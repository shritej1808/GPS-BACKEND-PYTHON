from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone, timedelta
from typing import Optional
import math, time, os, json

from pymongo import MongoClient
from bson import ObjectId

import firebase_admin
from firebase_admin import credentials, firestore, db, messaging

import razorpay

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
PAYMENTS_COLLECTION = "payments"

mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[DB_NAME]
trips_col = mongo_db[TRIP_COLLECTION]
devices_col = mongo_db[DEVICE_COLLECTION]
registered_col = mongo_db[REGISTERED_VEHICLES_COLLECTION]
payments_col = mongo_db[PAYMENTS_COLLECTION]

# =====================================================
# RAZORPAY CONFIG (TEST MODE)
# =====================================================
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "rzp_test_RoIFjBw2KFvSQF")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "yowJmfA6uehhJ6q3YiwBLPvH")

# Webhook secret (from Razorpay Dashboard → Settings → Webhooks)
RAZORPAY_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "test_webhook_secret")

razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

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

# Directions
DIRECTION_FORWARD = "forward"   # Dharwad → Hubballi
DIRECTION_RETURN = "return"     # Hubballi → Dharwad

# =====================================================
# TIMEZONE + FORMAT HELPER
# =====================================================
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
    except Exception:
        return "Not recorded"


def normalize_vehicle_id(raw: str) -> str:
    """
    Remove whitespace/newlines and upper-case the vehicle ID.
    Fixes 'UP70GT1215\\n' vs 'UP70GT1215'.
    """
    return raw.strip().upper()


# =====================================================
# SESSION (SINGLE VEHICLE DEMO)
# Direction-aware
# =====================================================
session = {
    "vehicle_id": None,
    "ocr_detected": False,
    "tracking": False,
    "distance_m": 0.0,
    "last_t": None,
    "trip_start_time": None,
    "was_off_road": False,

    # Directional segment
    "direction": None,          # "forward" or "return"
    "start_lat": None,
    "start_lon": None,
    "end_lat": None,
    "end_lon": None,
    "entry_toll_name": None,
    "exit_toll_name": None,
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


class CreateOrderRequest(BaseModel):
    vehicle_id: str
    amount: float          # in rupees (e.g. 50.75)
    trip_id: Optional[str] = None


class CreateOrderResponse(BaseModel):
    order_id: str
    amount: int            # in paise
    currency: str
    key_id: str            # send to Android


class VerifyPaymentRequest(BaseModel):
    order_id: str
    payment_id: str
    signature: str
    vehicle_id: str
    amount: float
    trip_id: Optional[str] = None


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
        chord = math.sqrt(sum((P[i] - A[i]) ** 2 for i in range(3)))
        distance = R * 2 * math.asin(chord / 2)
        return 0.0, distance

    t = dot(AP, AB) / denom
    t = max(0.0, min(1.0, t))

    closest = (A[0] + AB[0] * t, A[1] + AB[1] * t, A[2] + AB[2] * t)

    dx = P[0] - closest[0]
    dy = P[1] - closest[1]
    dz = P[2] - closest[2]
    chord = math.sqrt(dx * dx + dy * dy + dz * dz)

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
def save_trip_to_db(vehicle_id, distance_m, miles, toll, start_time, end_time,
                    entry_toll, exit_toll, direction: Optional[str]):
    """
    Store raw datetimes in MongoDB (keep them as datetime).
    Formatting is done when sending API response.
    """
    clean_vid = normalize_vehicle_id(vehicle_id)
    doc = {
        "vehicle_id": clean_vid,
        "distance_m": round(distance_m, 2),
        "distance_mi": round(miles, 2),
        "toll": round(toll, 2),
        "entry_toll": entry_toll,
        "exit_toll": exit_toll,
        "direction": direction,
        "start_time": start_time,
        "end_time": end_time,
        "created_at": datetime.now(timezone.utc),
        "is_paid": False,     # default unpaid
    }
    result = trips_col.insert_one(doc)
    print(f"💾 Trip stored in MongoDB with _id={result.inserted_id} for {clean_vid}")


def get_trips_for_vehicle(vehicle_id: str):
    """
    Returns trips with formatted date+time strings for Android:
    - distance  (Double)
    - toll      (Double)
    - startTime (String)
    - endTime   (String)
    - is_paid   (Boolean)
    - direction (String)
    """
    clean_vid = normalize_vehicle_id(vehicle_id)
    cursor = (
        trips_col.find({"vehicle_id": clean_vid})
        .sort("created_at", -1)
        .limit(50)
    )

    trips = []
    for doc in cursor:
        trips.append(
            {
                "_id": str(doc.get("_id")),
                "distance": doc.get("distance_mi", doc.get("distance_m", 0.0)),
                "toll": doc.get("toll", 0.0),
                "startTime": format_dt_for_output(doc.get("start_time")),
                "endTime": format_dt_for_output(doc.get("end_time")),
                "entry_toll": doc.get("entry_toll"),
                "exit_toll": doc.get("exit_toll"),
                "direction": doc.get("direction"),
                "created_at": format_dt_for_output(doc.get("created_at")),
                "is_paid": doc.get("is_paid", False),
            }
        )
    return trips

# =====================================================
# FCM
# =====================================================
def push_notify(vehicle_id: str, title: str, body: str, data=None):
    clean_vid = normalize_vehicle_id(vehicle_id)
    doc = devices_col.find_one({"vehicle_id": clean_vid})
    if not doc or "fcm_token" not in doc:
        print("⚠ No FCM token for", clean_vid)
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
    clean_vid = normalize_vehicle_id(payload.vehicle_id)
    devices_col.update_one(
        {"vehicle_id": clean_vid},
        {
            "$set": {
                "vehicle_id": clean_vid,
                "phone_number": payload.phone_number,
                "owner_name": payload.owner_name,
                "updated_at": datetime.now(timezone.utc),
            },
            "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
        },
        upsert=True,
    )
    print(f"✅ Owner registered for {clean_vid} ({payload.phone_number})")
    return {"message": "Owner registered/updated"}


@app.post("/register_device")
def register_device(payload: RegisterDevice):
    clean_vid = normalize_vehicle_id(payload.vehicle_id)
    devices_col.update_one(
        {"vehicle_id": clean_vid},
        {
            "$set": {
                "vehicle_id": clean_vid,
                "fcm_token": payload.fcm_token,
                "updated_at": datetime.now(timezone.utc),
            },
            "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
        },
        upsert=True,
    )
    print(f"📱 Device token registered for {clean_vid}")
    return {"message": "Device registered"}


@app.post("/register_vehicle")
def register_vehicle(payload: RegisterVehicle):
    clean_vid = normalize_vehicle_id(payload.vehicle_id)
    registered_col.update_one(
        {"vehicle_id": clean_vid},
        {
            "$set": {
                "vehicle_id": clean_vid,
                "owner_name": payload.owner_name,
                "updated_at": datetime.now(timezone.utc),
            },
            "$setOnInsert": {"created_at": datetime.now(timezone.utc)},
        },
        upsert=True,
    )
    print(f"✔ Registered Vehicle: {clean_vid}")
    return {"message": "Vehicle registered for toll tracking"}

# =====================================================
# START TRIP (OCR)
# =====================================================
@app.post("/start_trip")
def start_trip(data: VehicleData):
    raw_vid = data.vehicle_id
    vid = normalize_vehicle_id(raw_vid)
    print("\n🔥 /start_trip called with:", raw_vid, "→ normalized:", vid)

    record = registered_col.find_one({"vehicle_id": vid})
    if not record:
        print(f"❌ OCR DETECTED UNREGISTERED VEHICLE → {vid}")
        return {
            "status": "error",
            "allowed": False,
            "message": f"Vehicle {vid} is NOT registered for toll tracking."
        }

    session.update(
        {
            "vehicle_id": vid,
            "ocr_detected": True,
            "trip_start_time": None,
            "tracking": False,
            "distance_m": 0.0,
            "last_t": None,
            "was_off_road": False,

            "direction": None,
            "start_lat": None,
            "start_lon": None,
            "end_lat": None,
            "end_lon": None,
            "entry_toll_name": None,
            "exit_toll_name": None,
        }
    )

    print("\n================ OCR AUTHORIZED ================")
    print(f"[{time.strftime('%H:%M:%S')}] Vehicle Authorized: {vid}")
    print("OCR Trigger: Trip will start when entering either toll (any direction)")
    print("==============================================\n")

    push_notify(
        vid,
        title="Authorized Plate Detected",
        body="Trip will begin once vehicle enters a toll gate.",
        data={"type": "OCR_DETECTED"}
    )

    try:
        realtime_db.child("vehicles") \
            .child(vid) \
            .child("commands") \
            .update({"start": True, "stop": False})
        print(f"📡 Firebase command → {vid} commands.start = True")
    except Exception as e:
        print("❌ Firebase command error:", e)

    push_notify(
        vid,
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
    vid = normalize_vehicle_id(data.vehicle_id)

    print("\n=============== RESET TRIP ==================")
    print(f"[{time.strftime('%H:%M:%S')}] Reset for: {vid}")
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

            "direction": None,
            "start_lat": None,
            "start_lon": None,
            "end_lat": None,
            "end_lon": None,
            "entry_toll_name": None,
            "exit_toll_name": None,
        }
    )

    try:
        realtime_db.child("vehicles").child(vid).child("commands").update({
            "start": False,
            "stop": True
        })
    except Exception as e:
        print("⚠ Could not reset commands on reset:", e)

    return {"message": "Trip reset", "distance_mi": 0.0}

# =====================================================
# UPDATE LOCATION (DIRECTION-AWARE)
# =====================================================
@app.post("/update_location")
async def update_location(request: Request):
    data = await request.json()

    raw_vid = data["vehicle_id"]
    vid = normalize_vehicle_id(raw_vid)
    lat = float(data["latitude"])
    lon = float(data["longitude"])
    accuracy = float(data.get("accuracy", 0.0))

    print("\n---------------- NEW GPS UPDATE ----------------")
    print(f"Vehicle: {raw_vid} → {vid}")
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
        # Decide direction based on which toll we are closer to
        if d_entry <= ENTRY_RADIUS_M:
            # Forward: Dharwad -> Hubballi
            session["tracking"] = True
            session["distance_m"] = 0.0
            session["trip_start_time"] = datetime.now(timezone.utc)
            session["was_off_road"] = False

            session["direction"] = DIRECTION_FORWARD
            session["start_lat"] = ENTRY_LAT
            session["start_lon"] = ENTRY_LON
            session["end_lat"] = EXIT_LAT
            session["end_lon"] = EXIT_LON
            session["entry_toll_name"] = ENTRY_TOLL_NAME
            session["exit_toll_name"] = EXIT_TOLL_NAME

            t, _ = project_to_segment(lat, lon,
                                      session["start_lat"], session["start_lon"],
                                      session["end_lat"], session["end_lon"])
            session["last_t"] = t

            event = "🟢 Trip started (Dharwad → Hubballi)"
            print("\n🟢 ENTERED ENTRY TOLL → Tracking started (forward)")
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
                "Vehicle has entered the toll road (Dharwad → Hubballi).",
                {"type": "TRIP_STARTED", "direction": DIRECTION_FORWARD}
            )

            try:
                realtime_db.child("vehicles").child(vid).child("status").set("ON_ROAD")
            except Exception as e:
                print("⚠ Realtime DB status error:", e)

        elif d_exit <= ENTRY_RADIUS_M:
            # Return: Hubballi -> Dharwad
            session["tracking"] = True
            session["distance_m"] = 0.0
            session["trip_start_time"] = datetime.now(timezone.utc)
            session["was_off_road"] = False

            session["direction"] = DIRECTION_RETURN
            session["start_lat"] = EXIT_LAT
            session["start_lon"] = EXIT_LON
            session["end_lat"] = ENTRY_LAT
            session["end_lon"] = ENTRY_LON
            session["entry_toll_name"] = EXIT_TOLL_NAME
            session["exit_toll_name"] = ENTRY_TOLL_NAME

            t, _ = project_to_segment(lat, lon,
                                      session["start_lat"], session["start_lon"],
                                      session["end_lat"], session["end_lon"])
            session["last_t"] = t

            event = "🟢 Trip started (Hubballi → Dharwad)"
            print("\n🟢 ENTERED EXIT TOLL → Tracking started (return)")
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
                "Vehicle has entered the toll road (Hubballi → Dharwad).",
                {"type": "TRIP_STARTED", "direction": DIRECTION_RETURN}
            )

            try:
                realtime_db.child("vehicles").child(vid).child("status").set("ON_ROAD")
            except Exception as e:
                print("⚠ Realtime DB status error:", e)

        else:
            event = "Waiting near toll"
            print("⏳ Not inside any toll radius yet")

    # TRACKING
    else:
        print("🚗 Tracking active...")

        if session["start_lat"] is None or session["end_lat"] is None:
            print("⚠ Session missing start/end lat/lon → cannot project")
            event = "Tracking error"
        else:
            t, corridor_distance = project_to_segment(
                lat, lon,
                session["start_lat"], session["start_lon"],
                session["end_lat"], session["end_lon"]
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

            # Decide "end toll" depending on direction
            direction = session.get("direction")
            if direction == DIRECTION_FORWARD:
                # Expect to reach Hubballi Toll
                at_end = (d_exit <= EXIT_RADIUS_M and t >= 0.95)
            elif direction == DIRECTION_RETURN:
                # Expect to reach Dharwad Toll
                at_end = (d_entry <= EXIT_RADIUS_M and t >= 0.95)
            else:
                at_end = False

            if at_end:
                event = "🔴 Trip ended"
                print("\n🔴 END TOLL REACHED → Trip ended")

                distance = session["distance_m"]
                miles = distance * METER_TO_MILE
                toll = round(miles * TOLL_RATE_PER_MILE, 2)
                start_time = session.get("trip_start_time")
                end_time = datetime.now(timezone.utc)

                entry_toll_name = session.get("entry_toll_name") or ENTRY_TOLL_NAME
                exit_toll_name = session.get("exit_toll_name") or EXIT_TOLL_NAME

                if session["vehicle_id"]:
                    save_trip_to_db(
                        vehicle_id=session["vehicle_id"],
                        distance_m=distance,
                        miles=miles,
                        toll=toll,
                        start_time=start_time,
                        end_time=end_time,
                        entry_toll=entry_toll_name,
                        exit_toll=exit_toll_name,
                        direction=direction,
                    )

                try:
                    trip_id = f"{int(time.time())}"
                    firestore_db.collection("trips").document(vid).collection("all") \
                        .document(trip_id).set({
                            "vehicle_id": vid,
                            "distance_m": distance,
                            "toll": toll,
                            "start_time": format_dt_for_output(start_time),
                            "end_time": format_dt_for_output(end_time),
                            "entry_toll": entry_toll_name,
                            "exit_toll": exit_toll_name,
                            "direction": direction,
                            "created_at": format_dt_for_output(datetime.now(timezone.utc)),
                        })
                    print("📝 Trip summary written to Firestore (formatted times)")
                except Exception as e:
                    print("⚠ Firestore error:", e)

                push_notify(
                    vid,
                    "Trip Completed",
                    f"Toll: ₹{toll}",
                    {"type": "TRIP_ENDED", "direction": direction or ""}
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

                        "direction": None,
                        "start_lat": None,
                        "start_lon": None,
                        "end_lat": None,
                        "end_lon": None,
                        "entry_toll_name": None,
                        "exit_toll_name": None,
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
# RAZORPAY PAYMENT APIs (APP FLOW)
# =====================================================
@app.post("/create_order", response_model=CreateOrderResponse)
def create_order(payload: CreateOrderRequest):
    amount_rupees = payload.amount
    amount_paise = int(round(amount_rupees * 100))

    if amount_paise <= 0:
        raise HTTPException(status_code=400, detail="Amount must be > 0")

    try:
        order = razorpay_client.order.create({
            "amount": amount_paise,
            "currency": "INR",
            "payment_capture": 1,
            "notes": {
                "vehicle_id": normalize_vehicle_id(payload.vehicle_id),
                "trip_id": payload.trip_id or "",
            },
        })
    except Exception as e:
        print("❌ Razorpay order error:", e)
        raise HTTPException(status_code=500, detail="Failed to create order")

    payments_col.insert_one({
        "order_id": order["id"],
        "vehicle_id": normalize_vehicle_id(payload.vehicle_id),
        "trip_id": payload.trip_id,
        "amount_rupees": amount_rupees,
        "amount_paise": amount_paise,
        "currency": "INR",
        "status": "created",
        "created_at": datetime.now(timezone.utc),
    })

    print(f"💰 Razorpay order created: {order['id']} for ₹{amount_rupees:.2f}")

    return CreateOrderResponse(
        order_id=order["id"],
        amount=order["amount"],
        currency=order["currency"],
        key_id=RAZORPAY_KEY_ID,
    )


@app.post("/verify_payment")
def verify_payment(payload: VerifyPaymentRequest):
    """
    SIMPLE DEV-ONLY ENDPOINT:
    - No signature check
    - Just records payment + marks trip as paid.
    Real security is in `/razorpay_webhook`.
    """
    clean_vid = normalize_vehicle_id(payload.vehicle_id)

    payments_col.update_one(
        {"order_id": payload.order_id},
        {
            "$set": {
                "status": "paid_app_side",
                "payment_id": payload.payment_id,
                "vehicle_id": clean_vid,
                "amount_rupees": payload.amount,
                "trip_id": payload.trip_id,
                "verified_at": datetime.now(timezone.utc),
            }
        },
        upsert=True,
    )

    # Mark the trip as paid (best-effort)
    if payload.trip_id:
        try:
            trips_col.update_one(
                {"_id": ObjectId(payload.trip_id)},
                {"$set": {"is_paid": True}},
            )
            print(f"✔ Trip {payload.trip_id} marked as paid (from /verify_payment)")
        except Exception as e:
            print("⚠ Trip mark paid error:", e)

    print(f"✔ Payment saved (NO signature check). Order={payload.order_id}")

    return {
        "status": "success",
        "message": "Payment recorded (dev mode, no signature verification)",
        "order_id": payload.order_id,
        "payment_id": payload.payment_id,
    }

# =====================================================
# RAZORPAY WEBHOOK (REAL SIGNATURE CHECK)
# =====================================================
@app.post("/razorpay_webhook")
async def razorpay_webhook(request: Request):
    """
    Razorpay → backend webhook.
    - Verifies signature using RAZORPAY_WEBHOOK_SECRET
    - Updates payments collection
    - Marks trip.is_paid = True using trip_id from notes
    """
    body_bytes = await request.body()
    body_str = body_bytes.decode("utf-8")
    signature = request.headers.get("X-Razorpay-Signature")

    if not signature:
        print("❌ Missing X-Razorpay-Signature header")
        raise HTTPException(status_code=400, detail="Missing signature")

    # 1) Verify webhook signature
    try:
        razorpay.Utility.verify_webhook_signature(
            body_str,
            signature,
            RAZORPAY_WEBHOOK_SECRET,
        )
    except razorpay.errors.SignatureVerificationError as e:
        print("❌ Webhook signature failed:", e)
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    print("✅ Webhook signature verified")

    # 2) Parse payload
    try:
        payload = json.loads(body_str)
    except Exception as e:
        print("❌ Webhook JSON parse error:", e)
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event = payload.get("event")
    print(f"🔔 Razorpay Webhook event: {event}")

    # We mainly care about payment.captured
    if event == "payment.captured":
        pay_entity = payload.get("payload", {}).get("payment", {}).get("entity", {})
        order_id = pay_entity.get("order_id")
        payment_id = pay_entity.get("id")
        amount_paise = pay_entity.get("amount", 0)
        amount_rupees = amount_paise / 100.0 if amount_paise else 0.0
        notes = pay_entity.get("notes", {}) or {}

        vehicle_id = normalize_vehicle_id(notes.get("vehicle_id", "UNKNOWN"))
        trip_id = notes.get("trip_id") or None

        payments_col.update_one(
            {"order_id": order_id},
            {
                "$set": {
                    "status": "paid",
                    "payment_id": payment_id,
                    "vehicle_id": vehicle_id,
                    "amount_rupees": amount_rupees,
                    "amount_paise": amount_paise,
                    "trip_id": trip_id,
                    "webhook_event": event,
                    "webhook_received_at": datetime.now(timezone.utc),
                }
            },
            upsert=True,
        )

        # Mark trip as paid
        if trip_id:
            try:
                trips_col.update_one(
                    {"_id": ObjectId(trip_id)},
                    {"$set": {"is_paid": True}},
                )
                print(f"💳 Trip {trip_id} marked as PAID via webhook")
            except Exception as e:
                print("⚠ Trip mark paid error in webhook:", e)

        return {"status": "ok"}

    # Ignore other events for now
    print("ℹ️ Webhook event ignored (not payment.captured)")
    return {"status": "ignored", "event": event}

# =====================================================
# TRIP HISTORY
# =====================================================
@app.get("/trip_history/{vehicle_id}")
def trip_history(vehicle_id: str):
    trips = get_trips_for_vehicle(vehicle_id)
    clean_vid = normalize_vehicle_id(vehicle_id)
    return {"vehicle_id": clean_vid, "trips": trips}
# =====================================================
# SESSION STATE (for OCR script reset)
# =====================================================
@app.get("/session_state")
def session_state():
    return {
        "tracking": session.get("tracking"),
        "ocr_detected": session.get("ocr_detected"),
        "vehicle_id": session.get("vehicle_id"),
        "distance_m": session.get("distance_m"),
        "direction": session.get("direction"),
        "last_t": session.get("last_t"),
    }


# =====================================================
# ROOT
# =====================================================
@app.get("/")
def root():
    return {"status": "Running", "tolls": list(TOLL_ZONES.keys())}