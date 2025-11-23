from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import math, time

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- CONFIG ----------
METER_TO_MILE = 0.000621371
TOLL_RATE_PER_MILE = 8.05  # Rs per mile

# Toll Locations (YOUR EXACT COORDINATES)
TOLL_ZONES = {
    "Dharwad Toll": {"lat": 15.398638008097812, "lon": 75.00258199173943},
    "Hubballi Toll": {"lat": 15.394808141546225, "lon": 75.00719857222845},
}

# We want trip: Dharwad Toll  ->  Hubballi Toll
ENTRY_TOLL_NAME = "Dharwad Toll"
EXIT_TOLL_NAME  = "Hubballi Toll"

ENTRY_RADIUS_M = 250.0   # start trip if within 250m of ENTRY toll
EXIT_RADIUS_M  = 250.0   # end trip if within 250m of EXIT toll

DISTANCE_NOISE_FILTER = 1.0      # ignore <1m
TELEPORT_THRESHOLD    = 5000.0   # ignore insane jumps >5km
ACCURACY_LIMIT        = 80.0     # ignore very bad GPS if >80m


# ---------- MEMORY SESSION ----------
session = {
    "vehicle_id": None,
    "tracking": False,       # True only BETWEEN entry and exit toll
    "distance_m": 0.0,
    "last_lat": None,
    "last_lon": None,
    "last_event": "Idle",
}


class VehicleData(BaseModel):
    vehicle_id: str


# ---------- UTILS ----------
def haversine(lat1, lon1, lat2, lon2):
    """Distance in meters between two geo-points."""
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def distance_to_toll(lat, lon, toll_name: str) -> float:
    z = TOLL_ZONES[toll_name]
    return haversine(lat, lon, z["lat"], z["lon"])


def nearest_zone(lat, lon):
    best = (None, float("inf"))
    for name, z in TOLL_ZONES.items():
        d = haversine(lat, lon, z["lat"], z["lon"])
        if d < best[1]:
            best = (name, d)
    return best  # (zone_name, distance_m)


# ---------- RESET ----------
@app.post("/reset_distance")
def reset_distance(data: VehicleData):
    session.update({
        "vehicle_id": data.vehicle_id,
        "tracking": False,
        "distance_m": 0.0,
        "last_lat": None,
        "last_lon": None,
        "last_event": "Trip Reset",
    })
    print(f"[{time.strftime('%H:%M:%S')}] 🔄 Trip reset for {data.vehicle_id}")
    return {"message": "Trip reset", "distance_mi": 0.0}


# ---------- UPDATE LOCATION ----------
@app.post("/update_location")
async def update_location(request: Request):
    data = await request.json()
    vid = data.get("vehicle_id", "Unknown")

    try:
        lat = float(data["latitude"])
        lon = float(data["longitude"])
    except Exception:
        return {"error": "Invalid coordinates"}

    accuracy = float(data.get("accuracy", 0.0) or 0.0)

    if session["vehicle_id"] is None:
        session["vehicle_id"] = vid

    # Distances to entry & exit toll
    d_entry = distance_to_toll(lat, lon, ENTRY_TOLL_NAME)
    d_exit  = distance_to_toll(lat, lon, EXIT_TOLL_NAME)

    # Also nearest zone (for UI/debug)
    zone_name, d_zone = nearest_zone(lat, lon)
    print(f"📍 {lat:.6f}, {lon:.6f} → nearest={zone_name} ({d_zone:.1f}m)")

    # Detect same GPS point
    if session["last_lat"] is not None and session["last_lon"] is not None:
        same_point = (
            round(lat, 6) == round(session["last_lat"], 6)
            and round(lon, 6) == round(session["last_lon"], 6)
        )
    else:
        same_point = False

    event = None

    # -------------------------
    # NOT TRACKING YET → WAIT FOR ENTRY TOLL
    # -------------------------
    if not session["tracking"]:
        if d_entry <= ENTRY_RADIUS_M:
            session["tracking"] = True
            session["distance_m"] = 0.0
            session["last_lat"], session["last_lon"] = lat, lon
            event = f"🟢 Trip started at {ENTRY_TOLL_NAME}"
        else:
            event = "Waiting near entry toll"

    # -------------------------
    # TRACKING → COUNT DISTANCE UNTIL EXIT TOLL
    # -------------------------
    else:
        # 1) Add step distance if valid
        if not same_point and session["last_lat"] is not None:
            step = haversine(session["last_lat"], session["last_lon"], lat, lon)

            # Filter noise and insane jumps
            if DISTANCE_NOISE_FILTER <= step <= TELEPORT_THRESHOLD:
                if accuracy == 0.0 or accuracy <= ACCURACY_LIMIT:
                    session["distance_m"] += step
                # Always move the last point forward
                session["last_lat"], session["last_lon"] = lat, lon
            else:
                # Ignore distance, but advance pointer to avoid compounding jump
                session["last_lat"], session["last_lon"] = lat, lon
        else:
            # update last point if it was None initially
            if session["last_lat"] is None:
                session["last_lat"], session["last_lon"] = lat, lon

        # 2) Check exit toll
        if d_exit <= EXIT_RADIUS_M:
            event = f"🔴 Trip ended at {EXIT_TOLL_NAME}"
            session["tracking"] = False
        else:
            event = f"On route {ENTRY_TOLL_NAME} → {EXIT_TOLL_NAME}"

    # ---------- OUTPUT ----------
    miles = session["distance_m"] * METER_TO_MILE
    toll_amount = miles * TOLL_RATE_PER_MILE

    session["last_event"] = event

    print(
        f"[{time.strftime('%H:%M:%S')}] {vid}: {event} | "
        f"tracking={session['tracking']} | total={session['distance_m']:.1f}m "
        f"({miles:.2f}mi) | toll=₹{toll_amount:.2f}"
    )

    return {
        "vehicle_id": vid,
        "event": event,
        "active_zone": ENTRY_TOLL_NAME if session["tracking"] else None,
        "tracking": session["tracking"],
        "nearest_zone": {"name": zone_name, "distance_m": round(d_zone, 1)},
        "total_distance_m": round(session["distance_m"], 2),
        "total_distance_mi": round(miles, 2),
        "toll_estimate": round(toll_amount, 2),
        "distance_to_entry_m": round(d_entry, 1),
        "distance_to_exit_m": round(d_exit, 1),
    }


@app.get("/")
def root():
    return {"status": "Backend OK", "zones": list(TOLL_ZONES.keys())}
