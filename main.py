from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta
from typing import Optional
import math, time, os
from pymongo import MongoClient
import firebase_admin
from firebase_admin import credentials, firestore, db, messaging

# =====================================================
# FIREBASE INIT
# =====================================================
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://gps-tracking-system-86cc0-default-rtdb.firebaseio.com/"
})

firestore_db = firestore.client()
realtime_db = db.reference("/")

# =====================================================
# FASTAPI APP + CORS
# =====================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# MONGODB CONFIG
# =====================================================
MONGO_URI = os.getenv(
    "MONGO_URI",
    "mongodb+srv://admin:12345@cluster0.k8axcum.mongodb.net/"
)
DB_NAME = "gps_db"
TRIP_COLLECTION = "trip_history"
DEVICE_COLLECTION = "device_registry"
REGISTERED_VEHICLES_COLLECTION = "registered_vehicles"

mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[DB_NAME]
trips_col = mongo_db[TRIP_COLLECTION]
devices_col = mongo_db[DEVICE_COLLECTION]
registered_col = mongo_db[REGISTERED_VEHICLES_COLLECTION]

# =====================================================
# TOLL / DISTANCE CONFIG
# =====================================================
METER_TO_MILE = 0.000621371
TOLL_RATE_PER_MILE = 8.05  # ₹ per mile

TOLL_ZONES = {
    "Dharwad Toll": {"lat": 15.398638008097812, "lon": 75.00258199173943},
    "Hubballi Toll": {"lat": 15.394808141546225, "lon": 75.00719857222845},
}

ENTRY_TOLL_NAME = "Dharwad Toll"
EXIT_TOLL_NAME = "Hubballi Toll"

ENTRY_RADIUS_M = 250.0
EXIT_RADIUS_M = 100.0

CORRIDOR_WIDTH_M = 100.0
TELEPORT_THRESHOLD = 5000.0
ACCURACY_LIMIT = 80.0

# =====================================================
# TIMEZONE + FORMAT HELPER
# =====================================================
# IST (Indian Standard Time)
IST = timezone(timedelta(hours=5, minutes=30))

def format_dt_for_output(value) -> Optional[str]:
    if not value:
        return "Not recorded"

    try:
        if isinstance(value, str):
            dt = datetime.fromisoformat(value)
        else:
            dt = value

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        dt_ist = dt.astimezone(IST)
        return dt_ist.strftime("%d %b %Y, %I:%M %p")
    except:
        return "Not recorded"
# =====================================================
# SESSION (SINGLE VEHICLE DEMO)
# =====================================================
session = {
    "vehicle_id": None,
    "ocr_detected": False,
    "tracking": False,
    "distance_m": 0.0,
    "last_t": None,
    "trip_start_time": None,
    "was_off_road": False,
}

# =====================================================
# MODELS
# =====================================================
class VehicleData(BaseModel):
    vehicle_id: str

class RegisterOwner(BaseModel):
    vehicle_id: str
    phone_number: str
    owner_name: Optional[str] = None

class RegisterDevice(BaseModel):
    vehicle_id: str
    fcm_token: str

class RegisterVehicle(BaseModel):
    vehicle_id: str
    owner_name: Optional[str] = None

# =====================================================
# GEO UTILS
# =====================================================
def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def distance_to_toll(lat, lon, name):
    z = TOLL_ZONES[name]
    return haversine(lat, lon, z["lat"], z["lon"])

def nearest_zone(lat, lon):
    best = (None, float("inf"))
    for name, z in TOLL_ZONES.items():
        d = haversine(lat, lon, z["lat"], z["lon"])
        if d < best[1]:
            best = (name, d)
    return best

def project_to_segment(lat, lon, lat1, lon1, lat2, lon2) -> tuple[float, float]:
    p_lat, p_lon = math.radians(lat), math.radians(lon)
    a_lat, a_lon = math.radians(lat1), math.radians(lon1)
    b_lat, b_lon = math.radians(lat2), math.radians(lon2)

    R = 6371000.0

    def to_xyz(lat_r, lon_r):
        return (
            math.cos(lat_r) * math.cos(lon_r),
            math.cos(lat_r) * math.sin(lon_r),
            math.sin(lat_r),
        )

    P = to_xyz(p_lat, p_lon)
    A = to_xyz(a_lat, a_lon)
    B = to_xyz(b_lat, b_lon)

    def dot(u, v): return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
    def sub(u, v): return (u[0]-v[0], u[1]-v[1], u[2]-v[2])

    AB = sub(B, A)
    AP = sub(P, A)

    denom = dot(AB, AB)
    if denom == 0:
        chord = math.sqrt(sum((P[i] - A[i])**2 for i in range(3)))
        distance = R * 2 * math.asin(chord / 2)
        return 0.0, distance

    t = dot(AP, AB) / denom
    t = max(0.0, min(1.0, t))

    closest = (A[0] + AB[0]*t, A[1] + AB[1]*t, A[2] + AB[2]*t)

    dx = P[0] - closest[0]
    dy = P[1] - closest[1]
    dz = P[2] - closest[2]
    chord = math.sqrt(dx*dx + dy*dy + dz*dz)

    distance = R * 2 * math.asin(chord / 2)
    return t, distance

ENTRY_LAT = TOLL_ZONES[ENTRY_TOLL_NAME]["lat"]
ENTRY_LON = TOLL_ZONES[ENTRY_TOLL_NAME]["lon"]
EXIT_LAT = TOLL_ZONES[EXIT_TOLL_NAME]["lat"]
EXIT_LON = TOLL_ZONES[EXIT_TOLL_NAME]["lon"]
SEGMENT_LENGTH_M = haversine(ENTRY_LAT, ENTRY_LON, EXIT_LAT, EXIT_LON)
PROGRESS_THRESHOLD = 0.0001

# =====================================================
# DB HELPERS
# =====================================================
def save_trip_to_db(vehicle_id, distance_m, miles, toll, start_time, end_time):
    """
    Store raw datetimes in MongoDB (keep them as datetime).
    Formatting is done when sending API response.
    """
    doc = {
        "vehicle_id": vehicle_id,
        "distance_m": round(distance_m, 2),
        "distance_mi": round(miles, 2),
        "toll": round(toll, 2),
        "entry_toll": ENTRY_TOLL_NAME,
        "exit_toll": EXIT_TOLL_NAME,
        "start_time": start_time,
        "end_time": end_time,
        "created_at": datetime.now(timezone.utc),
    }
    result = trips_col.insert_one(doc)
    print(f"💾 Trip stored in MongoDB with _id={result.inserted_id}")

def get_trips_for_vehicle(vehicle_id: str):
    """
    Returns trips with formatted date+time strings for Android:
    - distance  (Double)
    - toll      (Double)
    - startTime (String)
    - endTime   (String)
    """
    cursor = (
        trips_col.find({"vehicle_id": vehicle_id})
        .sort("created_at", -1)
        .limit(50)
    )

    trips = []
    for doc in cursor:
        trips.append(
            {
                # Android expects these:
                "distance": doc.get("distance_mi", doc.get("distance_m", 0.0)),
                "toll": doc.get("toll", 0.0),
                "startTime": format_dt_for_output(doc.get("start_time")),
                "endTime": format_dt_for_output(doc.get("end_time")),

                # Extras (ignored by current Android model but useful later)
                "entry_toll": doc.get("entry_toll"),
                "exit_toll": doc.get("exit_toll"),
                "created_at": format_dt_for_output(doc.get("created_at")),
                "_id": str(doc.get("_id")),
            }
        )
    return trips

# =====================================================
# FCM
# =====================================================
def push_notify(vehicle_id: str, title: str, body: str, data=None):
    doc = devices_col.find_one({"vehicle_id": vehicle_id})
    if not doc or "fcm_token" not in doc:
        print("⚠ No FCM token for", vehicle_id)
        return

    token = doc["fcm_token"]

    message = messaging.Message(
        notification=messaging.Notification(title=title, body=body),
        data=data or {},
        token=token
    )

    try:
        response = messaging.send(message)
        print("📩 FCM Sent:", response)
    except Exception as e:
        print("❌ FCM Error:", e)

# =====================================================
# OWNER / DEVICE / VEHICLE REGISTRATION
# =====================================================
@app.post("/register_owner")
def register_owner(payload: RegisterOwner):
    devices_col.update_one(
        {"vehicle_id": payload.vehicle_id},
        {
            "$set": {
                "vehicle_id": payload.vehicle_id,
                "phone_number": payload.phone_number,
                "owner_name": payload.owner_name,
                "updated_at": datetime.now(timezone.utc),
            },
            "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
        },
        upsert=True,
    )
    print(f"✅ Owner registered for {payload.vehicle_id} ({payload.phone_number})")
    return {"message": "Owner registered/updated"}

@app.post("/register_device")
def register_device(payload: RegisterDevice):
    devices_col.update_one(
        {"vehicle_id": payload.vehicle_id},
        {
            "$set": {
                "vehicle_id": payload.vehicle_id,
                "fcm_token": payload.fcm_token,
                "updated_at": datetime.now(timezone.utc),
            },
            "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
        },
        upsert=True,
    )
    print(f"📱 Device token registered for {payload.vehicle_id}")
    return {"message": "Device registered"}

@app.post("/register_vehicle")
def register_vehicle(payload: RegisterVehicle):
    registered_col.update_one(
        {"vehicle_id": payload.vehicle_id},
        {
            "$set": {
                "vehicle_id": payload.vehicle_id,
                "owner_name": payload.owner_name,
                "updated_at": datetime.now(timezone.utc),
            },
            "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
        },
        upsert=True,
    )
    print(f"✔ Registered Vehicle: {payload.vehicle_id}")
    return {"message": "Vehicle registered for toll tracking"}

# =====================================================
# START TRIP (OCR)
# =====================================================
@app.post("/start_trip")
def start_trip(data: VehicleData):
    print("\n🔥 /start_trip called with:", data.vehicle_id)

    record = registered_col.find_one({"vehicle_id": data.vehicle_id})
    if not record:
        print(f"❌ OCR DETECTED UNREGISTERED VEHICLE → {data.vehicle_id}")
        return {
            "status": "error",
            "allowed": False,
            "message": f"Vehicle {data.vehicle_id} is NOT registered for toll tracking."
        }

    session.update(
        {
            "vehicle_id": data.vehicle_id,
            "ocr_detected": True,
            "trip_start_time": None,
            "tracking": False,
            "distance_m": 0.0,
            "last_t": None,
            "was_off_road": False,
        }
    )

    print("\n================ OCR AUTHORIZED ================")
    print(f"[{time.strftime('%H:%M:%S')}] Vehicle Authorized: {data.vehicle_id}")
    print("OCR Trigger: Trip will start when entering entry toll")
    print("==============================================\n")

    push_notify(
        data.vehicle_id,
        title="Authorized Plate Detected",
        body="Trip will begin once vehicle enters the entry toll.",
        data={"type": "OCR_DETECTED"}
    )

    try:
        realtime_db.child("vehicles") \
            .child(data.vehicle_id) \
            .child("commands") \
            .update({"start": True, "stop": False})
        print(f"📡 Firebase command → {data.vehicle_id} commands.start = True")
    except Exception as e:
        print("❌ Firebase command error:", e)

    push_notify(
        data.vehicle_id,
        title="Start Tracking",
        body="OCR detected. Begin sending GPS...",
        data={"type": "START_GPS"}
    )

    return {
        "status": "success",
        "allowed": True,
        "message": "Authorized plate detected. Mobile app triggered to start GPS."
    }

# =====================================================
# RESET TRIP
# =====================================================
@app.post("/reset_distance")
def reset_distance(data: VehicleData):
    print("\n=============== RESET TRIP ==================")
    print(f"[{time.strftime('%H:%M:%S')}] Reset for: {data.vehicle_id}")
    print("============================================\n")

    session.update(
        {
            "tracking": False,
            "ocr_detected": False,
            "distance_m": 0.0,
            "last_t": None,
            "trip_start_time": None,
            "was_off_road": False,
            "vehicle_id": None,
        }
    )

    try:
        realtime_db.child("vehicles").child(data.vehicle_id).child("commands").update({
            "start": False,
            "stop": True
        })
    except Exception as e:
        print("⚠ Could not reset commands on reset:", e)

    return {"message": "Trip reset", "distance_mi": 0.0}

# =====================================================
# UPDATE LOCATION (same logic as your latest code, but
#   FIRESTORE write now uses formatted date+time)
# =====================================================
@app.post("/update_location")
async def update_location(request: Request):
    data = await request.json()

    vid = data["vehicle_id"]
    lat = float(data["latitude"])
    lon = float(data["longitude"])
    accuracy = float(data.get("accuracy", 0.0))

    print("\n---------------- NEW GPS UPDATE ----------------")
    print(f"Vehicle: {vid}")
    print(f"Lat: {lat}, Lon: {lon}, Accuracy: {accuracy}m")

    try:
        realtime_db.child("vehicles").child(vid).child("location").set({
            "lat": lat,
            "lon": lon,
            "accuracy": accuracy,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    except Exception as e:
        print("⚠ Realtime DB error:", e)

    d_entry = distance_to_toll(lat, lon, ENTRY_TOLL_NAME)
    d_exit = distance_to_toll(lat, lon, EXIT_TOLL_NAME)
    zone_name, d_zone = nearest_zone(lat, lon)

    print(f"Nearest Zone: {zone_name} ({round(d_zone, 1)}m)")
    print(f"Distance to ENTRY: {round(d_entry, 1)}m | EXIT: {round(d_exit, 1)}m")

    if accuracy > ACCURACY_LIMIT:
        print(f"⚠ Poor accuracy {accuracy}m > {ACCURACY_LIMIT}m → Update ignored")
        miles = session["distance_m"] * METER_TO_MILE
        toll = miles * TOLL_RATE_PER_MILE
        return {
            "event": "Poor GPS accuracy",
            "tracking": session["tracking"],
            "nearest_zone": {"name": zone_name, "distance_m": round(d_zone, 1)},
            "total_distance_mi": round(miles, 2),
            "toll_estimate": round(toll, 2),
        }

    # BEFORE TRIP START
    if not session["ocr_detected"]:
        event = "Waiting for number plate detection"
        print("❌ OCR NOT DETECTED → Cannot track yet")

    elif not session["tracking"]:
        if d_entry <= ENTRY_RADIUS_M:
            t, _ = project_to_segment(lat, lon, ENTRY_LAT, ENTRY_LON, EXIT_LAT, EXIT_LON)

            session["tracking"] = True
            session["distance_m"] = 0.0
            session["last_t"] = t
            session["trip_start_time"] = datetime.now(timezone.utc)
            session["was_off_road"] = False

            event = "🟢 Trip started"
            print("\n🟢 ENTERED ENTRY TOLL → Tracking started")
            print(f"Initial t: {t:.4f}")

            try:
                realtime_db.child("vehicles").child(vid).child("commands").update({
                    "start": False
                })
                print("🧹 Cleared start trigger (trip started)")
            except Exception as e:
                print("⚠ Could not reset start trigger:", e)

            push_notify(
                vid,
                "Trip Started",
                "Vehicle has entered the toll road.",
                {"type": "TRIP_STARTED"}
            )

            try:
                realtime_db.child("vehicles").child(vid).child("status").set("ON_ROAD")
            except Exception as e:
                print("⚠ Realtime DB status error:", e)

        else:
            event = "Waiting near entry toll"
            print("⏳ Not inside ENTRY toll radius yet")

    # TRACKING
    else:
        print("🚗 Tracking active...")

        t, corridor_distance = project_to_segment(
            lat, lon, ENTRY_LAT, ENTRY_LON, EXIT_LAT, EXIT_LON
        )

        print(
            f"Corridor Distance: {corridor_distance:.1f}m "
            f"(limit={CORRIDOR_WIDTH_M}m)"
        )
        print(f"Projected t: {t:.4f}")

        if corridor_distance > CORRIDOR_WIDTH_M:
            print("🚫 OFF the toll road → distance ignored")
            event = "Off toll road"

            if not session.get("was_off_road", False):
                push_notify(
                    vid,
                    "Off Toll Route",
                    "Vehicle left the toll road!",
                    {"type": "OFF_ROAD"}
                )
                session["was_off_road"] = True

            try:
                realtime_db.child("vehicles").child(vid).child("status").set("OFF_ROAD")
            except Exception as e:
                print("⚠ Realtime DB status error:", e)

            miles = session["distance_m"] * METER_TO_MILE
            toll = miles * TOLL_RATE_PER_MILE

            print(
                f"TOTAL DISTANCE (unchanged): {session['distance_m']:.2f} m "
                f"({miles:.2f} mi)"
            )
            print(f"ESTIMATED TOLL: ₹{toll:.2f}")
            print("------------------------------------------------\n")

            return {
                "event": event,
                "tracking": True,
                "nearest_zone": {"name": zone_name, "distance_m": round(d_zone, 1)},
                "total_distance_mi": round(miles, 2),
                "toll_estimate": round(toll, 2),
            }

        print("🛣 ON toll road corridor → counting movement")

        if session.get("was_off_road", False):
            push_notify(
                vid,
                "Back On Route",
                "Vehicle returned to toll road.",
                {"type": "ON_ROAD"}
            )
            session["was_off_road"] = False

        try:
            realtime_db.child("vehicles").child(vid).child("status").set("ON_ROAD")
        except Exception as e:
            print("⚠ Realtime DB status error:", e)

        if session["last_t"] is not None:
            delta_t = t - session["last_t"]

            if delta_t < 0:
                print(f"⚠ Backward movement ignored: delta_t={delta_t:.4f}")
                delta_t = 0.0

            step_m = delta_t * SEGMENT_LENGTH_M

            if delta_t > PROGRESS_THRESHOLD:
                if step_m > TELEPORT_THRESHOLD:
                    print(f"⚠ Ignored step: Teleport along corridor ({step_m:.2f}m)")
                else:
                    session["distance_m"] += step_m
                    print(
                        f"✔ Progress added: delta_t={delta_t:.4f}, "
                        f"step={step_m:.2f} m"
                    )
                    print(f"✔ Total distance: {session['distance_m']:.2f} m")
            else:
                print("⚠ No significant forward progress")
        else:
            print("⚠ Initializing last_t during tracking")

        session["last_t"] = t

        if d_exit <= EXIT_RADIUS_M and t >= 0.95:
            event = "🔴 Trip ended"
            print("\n🔴 EXIT TOLL REACHED → Trip ended")

            distance = session["distance_m"]
            miles = distance * METER_TO_MILE
            toll = round(miles * TOLL_RATE_PER_MILE, 2)
            start_time = session.get("trip_start_time")
            end_time = datetime.now(timezone.utc)

            if session["vehicle_id"]:
                save_trip_to_db(
                    vehicle_id=session["vehicle_id"],
                    distance_m=distance,
                    miles=miles,
                    toll=toll,
                    start_time=start_time,
                    end_time=end_time,
                )

            # -------- Firestore: store formatted date+time strings ----------
            try:
                trip_id = f"{int(time.time())}"
                firestore_db.collection("trips").document(vid).collection("all") \
                    .document(trip_id).set({
                        "vehicle_id": vid,
                        "distance_m": distance,
                        "toll": toll,
                        "start_time": format_dt_for_output(start_time),
                        "end_time": format_dt_for_output(end_time),
                        "entry_toll": ENTRY_TOLL_NAME,
                        "exit_toll": EXIT_TOLL_NAME,
                        "created_at": format_dt_for_output(datetime.now(timezone.utc)),
                    })
                print("📝 Trip summary written to Firestore (formatted times)")
            except Exception as e:
                print("⚠ Firestore error:", e)
            # ----------------------------------------------------------------

            push_notify(
                vid,
                "Trip Completed",
                f"Toll: ₹{toll}",
                {"type": "TRIP_ENDED"}
            )

            try:
                realtime_db.child("vehicles").child(vid).child("commands").update({
                    "start": False,
                    "stop": True
                })
                print(f"📡 STOP command sent to {vid}")
            except Exception as e:
                print("⚠ STOP trigger error:", e)

            session.update(
                {
                    "tracking": False,
                    "ocr_detected": False,
                    "last_t": None,
                    "was_off_road": False,
                    "vehicle_id": None,
                }
            )
        else:
            event = "Moving inside toll route"

    miles = session["distance_m"] * METER_TO_MILE
    toll = miles * TOLL_RATE_PER_MILE

    print(f"TOTAL DISTANCE: {session['distance_m']:.2f} m ({miles:.2f} mi)")
    print(f"ESTIMATED TOLL: ₹{toll:.2f}")
    print("------------------------------------------------\n")

    return {
        "event": event,
        "tracking": session["tracking"],
        "nearest_zone": {"name": zone_name, "distance_m": round(d_zone, 1)},
        "total_distance_mi": round(miles, 2),
        "toll_estimate": round(toll, 2),
    }

# =====================================================
# TRIP HISTORY
# =====================================================
@app.get("/trip_history/{vehicle_id}")
def trip_history(vehicle_id: str):
    trips = get_trips_for_vehicle(vehicle_id)
    return {"vehicle_id": vehicle_id, "trips": trips}

# =====================================================
# ROOT
# =====================================================
@app.get("/")
def root():
    return {"status": "Running", "tolls": list(TOLL_ZONES.keys())}
