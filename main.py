from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import math, time
from datetime import datetime

from pymongo import MongoClient
from bson.objectid import ObjectId

app = FastAPI()

# -----------------------------------------------
# CORS
# -----------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------
# MONGODB CONFIG
# -----------------------------------------------
MONGO_URI = "mongodb+srv://admin:12345@cluster0.k8axcum.mongodb.net/"
DB_NAME = "gps_db"
TRIP_COLLECTION = "trip_history"

mongo_client = MongoClient(MONGO_URI)
mongo_db = mongo_client[DB_NAME]
trips_col = mongo_db[TRIP_COLLECTION]

# -----------------------------------------------
# CONFIG
# -----------------------------------------------
METER_TO_MILE = 0.000621371
TOLL_RATE_PER_MILE = 8.05

TOLL_ZONES = {
    "Dharwad Toll": {"lat": 15.398638008097812, "lon": 75.00258199173943},
    "Hubballi Toll": {"lat": 15.394808141546225, "lon": 75.00719857222845},
}

ENTRY_TOLL_NAME = "Dharwad Toll"
EXIT_TOLL_NAME = "Hubballi Toll"

ENTRY_RADIUS_M = 250.0
EXIT_RADIUS_M = 250.0

DISTANCE_NOISE_FILTER = 1.0
TELEPORT_THRESHOLD = 5000.0
ACCURACY_LIMIT = 80.0

# -----------------------------------------------
# SESSION
# -----------------------------------------------
session = {
    "vehicle_id": None,
    "ocr_detected": False,
    "tracking": False,
    "distance_m": 0.0,
    "last_lat": None,
    "last_lon": None,
    "trip_start_time": None,   # string like "12:34:56"
}


class VehicleData(BaseModel):
    vehicle_id: str


# -----------------------------------------------
# UTILS
# -----------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Distance in meters between two geo points."""
    R = 6371000.0
    p1, p2 = map(math.radians, [lat1, lat2])
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    )
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


# -----------------------------------------------
# SAVE TRIP TO MONGO
# -----------------------------------------------
def save_trip_to_db(
    vehicle_id: str,
    distance_m: float,
    miles: float,
    toll: float,
    start_time: str | None,
    end_time: str | None,
):
    doc = {
        "vehicle_id": vehicle_id,
        "distance_m": round(distance_m, 2),
        "distance_mi": round(miles, 2),
        "toll": round(toll, 2),
        "entry_toll": ENTRY_TOLL_NAME,
        "exit_toll": EXIT_TOLL_NAME,
        "start_time": start_time,
        "end_time": end_time,
        "created_at": datetime.utcnow(),
    }
    result = trips_col.insert_one(doc)
    print(f"💾 Trip stored in MongoDB with _id={result.inserted_id}")


# -----------------------------------------------
# OCR TRIGGER
# -----------------------------------------------
@app.post("/start_trip")
def start_trip(data: VehicleData):
    session.update(
        {
            "vehicle_id": data.vehicle_id,
            "ocr_detected": True,
            "trip_start_time": None,
        }
    )

    print("\n================ OCR DETECTED ================")
    print(f"[{time.strftime('%H:%M:%S')}] Vehicle: {data.vehicle_id}")
    print("OCR Trigger: Trip will start when entering entry toll")
    print("==============================================\n")

    return {"message": "Plate detected. Trip will start when entering toll."}


# -----------------------------------------------
# RESET TRIP
# -----------------------------------------------
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
            "last_lat": None,
            "last_lon": None,
            "trip_start_time": None,
        }
    )

    return {"message": "Trip reset", "distance_mi": 0.0}


# -----------------------------------------------
# UPDATE LOCATION
# -----------------------------------------------
@app.post("/update_location")
async def update_location(request: Request):
    data = await request.json()

    vid = data["vehicle_id"]
    lat = float(data["latitude"])
    lon = float(data["longitude"])
    accuracy = float(data.get("accuracy", 0.0))

    # ensure vehicle_id stored at least once
    if session["vehicle_id"] is None:
        session["vehicle_id"] = vid

    print("\n---------------- NEW GPS UPDATE ----------------")
    print(f"Vehicle: {vid}")
    print(f"Lat: {lat}, Lon: {lon}, Accuracy: {accuracy}m")

    d_entry = distance_to_toll(lat, lon, ENTRY_TOLL_NAME)
    d_exit = distance_to_toll(lat, lon, EXIT_TOLL_NAME)
    zone_name, d_zone = nearest_zone(lat, lon)

    print(f"Nearest Zone: {zone_name} ({round(d_zone, 1)}m)")
    print(f"Distance to ENTRY: {round(d_entry, 1)}m | EXIT: {round(d_exit, 1)}m")

    same_point = (
        session["last_lat"] is not None
        and round(lat, 6) == round(session["last_lat"], 6)
        and round(lon, 6) == round(session["last_lon"], 6)
    )

    # ---------------------------------------
    # BEFORE TRIP START
    # ---------------------------------------
    if not session["ocr_detected"]:
        event = "Waiting for number plate detection"
        print("❌ OCR NOT DETECTED → Cannot track yet")

    elif not session["tracking"]:
        if d_entry <= ENTRY_RADIUS_M:
            session["tracking"] = True
            session["distance_m"] = 0.0
            session["last_lat"], session["last_lon"] = lat, lon
            session["trip_start_time"] = time.strftime("%H:%M:%S")

            event = "🟢 Trip started"
            print("\n🟢 ENTERED ENTRY TOLL → Tracking started")
            print(f"Trip start time: {session['trip_start_time']}")
        else:
            event = "Waiting near entry toll"
            print("⏳ Not inside ENTRY toll radius")

    else:
        # ---------------------------------------
        # TRACKING MODE
        # ---------------------------------------
        print("🚗 Tracking active...")

        if not same_point:
            step = haversine(session["last_lat"], session["last_lon"], lat, lon)
            print(f"Raw Step Distance: {step:.2f} m")

            if step < DISTANCE_NOISE_FILTER:
                print("⚠ Ignored step: Too small / noise")
            elif step > TELEPORT_THRESHOLD:
                print("⚠ Ignored step: Teleport (GPS jump)")
            elif accuracy > ACCURACY_LIMIT:
                print(f"⚠ Ignored step: Poor accuracy {accuracy}m > {ACCURACY_LIMIT}m")
            else:
                session["distance_m"] += step
                print(f"✔ Step added: {step:.2f} m")
                print(f"✔ Total distance: {session['distance_m']:.2f} m")

            session["last_lat"], session["last_lon"] = lat, lon
        else:
            print("⚠ GPS same location → No movement")

        # EXIT DETECTION
        if d_exit <= EXIT_RADIUS_M:
            event = "🔴 Trip ended"
            print("\n🔴 EXIT TOLL REACHED → Trip ended")

            distance = session["distance_m"]
            miles = distance * METER_TO_MILE
            toll = miles * TOLL_RATE_PER_MILE
            end_time = time.strftime("%H:%M:%S")

            # SAVE TO MONGO HERE 🔥
            save_trip_to_db(
                vehicle_id=session["vehicle_id"] or vid,
                distance_m=distance,
                miles=miles,
                toll=toll,
                start_time=session["trip_start_time"],
                end_time=end_time,
            )

            session.update(
                {
                    "tracking": False,
                    "ocr_detected": False,
                    "distance_m": 0.0,          # optional: reset distance after saving
                    "last_lat": None,
                    "last_lon": None,
                    "trip_start_time": None,
                }
            )
        else:
            event = "Moving inside toll route"

    miles = session["distance_m"] * METER_TO_MILE
    toll = miles * TOLL_RATE_PER_MILE

    print(f"TOTAL DISTANCE: {session['distance_m']:.2f} m ({miles:.2f} mi)")
    print(f"ESTIMATED TOLL (live): ₹{toll:.2f}")
    print("------------------------------------------------\n")

    return {
        "event": event,
        "tracking": session["tracking"],
        "nearest_zone": {"name": zone_name, "distance_m": round(d_zone, 1)},
        "total_distance_mi": round(miles, 2),
        "toll_estimate": round(toll, 2),
        "start_time": session["trip_start_time"],
    }


# -----------------------------------------------
# TRIP HISTORY ENDPOINTS
# -----------------------------------------------
def serialize_trip(doc):
    return {
        "id": str(doc.get("_id")),
        "vehicle_id": doc.get("vehicle_id"),
        "distance_m": doc.get("distance_m", 0.0),
        "distance_mi": doc.get("distance_mi", 0.0),
        "toll": doc.get("toll", 0.0),
        "entry_toll": doc.get("entry_toll"),
        "exit_toll": doc.get("exit_toll"),
        "start_time": doc.get("start_time"),
        "end_time": doc.get("end_time"),
        "created_at": doc.get("created_at").isoformat()
        if doc.get("created_at")
        else None,
    }


@app.get("/trips")
def get_trips(vehicle_id: str | None = None, limit: int = 50):
    """
    GET /trips
    Optional: ?vehicle_id=UP70GT1215&limit=20
    """
    query = {}
    if vehicle_id:
        query["vehicle_id"] = vehicle_id

    cursor = trips_col.find(query).sort("created_at", -1).limit(limit)
    trips = [serialize_trip(doc) for doc in cursor]
    return {"count": len(trips), "trips": trips}


@app.get("/trips/{trip_id}")
def get_trip_by_id(trip_id: str):
    try:
        doc = trips_col.find_one({"_id": ObjectId(trip_id)})
        if not doc:
            return {"error": "Trip not found"}
        return serialize_trip(doc)
    except Exception:
        return {"error": "Invalid trip id"}
# -----------------------------------------------
# GET ALL TRIP HISTORY (READ FROM MONGO)
# -----------------------------------------------
@app.get("/trip_history")
def get_trip_history():
    trips = list(trips_col.find().sort("created_at", -1))  # newest first

    # Convert ObjectId to string
    for t in trips:
        t["_id"] = str(t["_id"])

    return {"count": len(trips), "data": trips}


# -----------------------------------------------
# ROOT
# -----------------------------------------------
@app.get("/")
def root():
    return {"status": "Running"}
