---

# 🚦 GPS Backend – Toll Tracking & Payment System (FastAPI)

This repository contains the **backend system** for a **GPS-based toll tracking application**.
It powers real-time vehicle tracking, toll calculation, trip history management, Firebase notifications, and secure payment processing using Razorpay.

The backend is designed to work seamlessly with an **Android GPS application** and an **OCR-based vehicle detection system**.

---

## 🧠 What This Backend Does

✔ Authorizes vehicles detected via OCR
✔ Starts/stops GPS tracking automatically
✔ Calculates distance & toll charges accurately
✔ Handles forward & return toll routes
✔ Stores trip history securely
✔ Sends real-time Firebase notifications
✔ Integrates Razorpay payments (with webhook verification)

---

## 🏗️ Architecture Overview

```
Android App  ──► FastAPI Backend ──► MongoDB
     │                │
     │                ├── Firebase (FCM + Realtime DB)
     │                │
     │                └── Razorpay (Payments & Webhooks)
     │
OCR System ──► /start_trip API
```

---

## 🚀 Features

### 🔐 Vehicle Authentication

* Vehicle-based login
* Prevents unregistered vehicles from starting trips
* OCR-based authorization flow

### 🛰️ GPS Tracking & Toll Logic

* Direction-aware tracking (Forward & Return)
* Haversine-based distance calculation
* Corridor validation to detect off-road movement
* Teleport & GPS accuracy filtering

### 🧭 Trip Lifecycle Management

* Automatic trip start at toll entry
* Automatic trip end at toll exit
* MongoDB trip history storage
* Firebase Firestore trip summaries

### 🔔 Firebase Integration

* Firebase Cloud Messaging (FCM)
* Realtime Database commands
* Push notifications for:

  * Trip start
  * Trip end
  * Off-road alerts
  * OCR authorization

### 💳 Payment Integration (Razorpay)

* Order creation
* Payment verification (App-side)
* Secure webhook signature validation
* Trip payment status updates

---

## 🧰 Tech Stack

| Layer             | Technology                   |
| ----------------- | ---------------------------- |
| Backend Framework | **FastAPI (Python)**         |
| Database          | **MongoDB**                  |
| Realtime Updates  | **Firebase Realtime DB**     |
| Notifications     | **Firebase Cloud Messaging** |
| Payments          | **Razorpay**                 |
| Auth Model        | Vehicle-based                |
| Timezone Handling | IST (UTC +5:30)              |

---

## 📁 Project Structure

```
GPS-BACKEND-PYTHON/
├── main.py                # FastAPI application
├── firebase_key.json      # Firebase service account key
├── requirements.txt       # Python dependencies
├── .gitignore
├── .qodo
└── __pycache__/
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone <your-backend-repo-url>
cd GPS-BACKEND-PYTHON
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Firebase Setup

* Place your **Firebase service account key** as:

  ```
  firebase_key.json
  ```
* Ensure Firebase Realtime DB & FCM are enabled

---

## 🔐 Environment Variables

You can set these as environment variables (recommended):

```bash
MONGO_URI=mongodb+srv://<username>:<password>@cluster.mongodb.net/
RAZORPAY_KEY_ID=rzp_test_xxxxx
RAZORPAY_KEY_SECRET=xxxxxxxx
RAZORPAY_WEBHOOK_SECRET=your_webhook_secret
```

(Default test values are present for development.)

---

## ▶️ Running the Server

```bash
uvicorn main:app --reload
```

Server will run at:

```
http://127.0.0.1:8000
```

---

## 📡 API Endpoints (Core)

### 🚗 Vehicle & Device

* `POST /register_vehicle`
* `POST /register_owner`
* `POST /register_device`
* `POST /check_vehicle`
* `POST /logout_vehicle`

### 🛰️ GPS & Trips

* `POST /start_trip`
* `POST /update_location`
* `POST /reset_distance`
* `GET  /trip_history/{vehicle_id}`

### 💳 Payments

* `POST /create_order`
* `POST /verify_payment`
* `POST /razorpay_webhook`

### 🔍 Debug

* `GET /session_state`
* `GET /get_logged_in_vehicle`

---

## 🧮 Toll Calculation Logic

* Distance calculated using **Haversine formula**
* Distance converted to miles
* Toll calculated as:

```
toll = distance_in_miles × rate_per_mile
```

Supports:

* Forward direction (Dharwad → Hubballi)
* Return direction (Hubballi → Dharwad)

---

## 🔔 Notifications Sent

* OCR authorized
* Start GPS tracking
* Trip started
* Off toll-road warning
* Trip completed
* Payment status updates

---

## 🔒 Security Notes

⚠ **Important**

* `/verify_payment` is **DEV MODE only**
* Real payment validation happens via `/razorpay_webhook`
* Signature verification is enforced for webhook

---

## 🔮 Future Enhancements

* JWT-based admin authentication
* Multi-toll corridor support
* Live trip dashboard
* Vehicle analytics
* Fraud detection
* Map visualization APIs

---

## 🎯 Why This Backend Is Strong

This backend demonstrates **real-world system design**:

* Event-driven tracking
* Stateful session control
* Payment security
* Cloud messaging
* Precision GPS logic

Perfect for **final-year projects, startups, or interviews**.

---


