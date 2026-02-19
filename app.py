# app.py
import os
import json
import cv2
from datetime import datetime, timedelta
from collections import Counter

from flask import Flask, render_template, request, redirect, session, url_for, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import numpy as np
from PIL import Image
from pymongo import MongoClient
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
app = Flask(__name__)
app.secret_key = "secret_key"

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client = MongoClient("mongodb://localhost:27017/")
db = client["soyabeen_prediction_db"]

users_collection = db["users"]
predictions_collection = db["predictions"]
feedback_collection = db["feedback"]

# ---------------- LOAD MODEL ----------------
model = load_model("model.h5")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

print("Loaded class names:", class_names)

# ---------------- IMAGE PREPROCESS (GRAYSCALE) ----------------
def preprocess_image(path, target_size=(150, 150)):
    img = Image.open(path).convert("L")
    img = img.resize(target_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ---------------- ROUTES ----------------

@app.route("/")
def index():
    if "username" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

# -------- LOGIN --------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = users_collection.find_one({"username": username})
        if user and check_password_hash(user["password_hash"], password):
            session["username"] = username
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials")

    return render_template("login.html")

# -------- REGISTER --------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users_collection.insert_one({
            "username": username,
            "password_hash": generate_password_hash(password)
        })

        flash("Registered successfully")
        return redirect(url_for("login"))

    return render_template("register.html")

# -------- DASHBOARD --------
@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))

    last_prediction = predictions_collection.find_one(
        {"username": session["username"]},
        sort=[("timestamp", -1)]
    )

    last_result = last_prediction["prediction"] if last_prediction else "No predictions yet"

    return render_template(
        "dashboard.html",
        farmer_name=session["username"],
        city_name="India",
        current_date=datetime.now().strftime("%d %b %Y"),
        last_prediction=last_result
    )

# -------- PROFILE --------
@app.route("/profile")
def profile():
    if "username" not in session:
        return redirect(url_for("login"))

    user = users_collection.find_one(
        {"username": session["username"]},
        {"_id": 0, "password_hash": 0}
    )

    return render_template("profile.html", user=user)

# -------- PAST REPORT --------
@app.route("/past-report")
def past_report():
    if "username" not in session:
        return redirect(url_for("login"))

    reports = list(predictions_collection.find(
        {"username": session["username"]}
    ).sort("timestamp", -1))

    return render_template("past_report.html", reports=reports)

# -------- CONTACT --------
@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        feedback_collection.insert_one({
            "name": request.form["name"],
            "email": request.form["email"],
            "message": request.form["message"],
            "timestamp": datetime.now()
        })
        flash("Message sent successfully")
        return redirect(url_for("contact"))

    return render_template("contact.html")

@app.route("/upload_img", methods=["GET", "POST"])
def upload_img():
    if "username" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        file = request.files["image"]

        if not file:
            flash("No image selected", "error")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        file.save(filepath)

        # ---------- PREDICTION ----------
        img_array = preprocess_image(filepath)
        prediction = model.predict(img_array)

        idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        predicted_class = class_names[idx]

        # ---------- DISEASE LOCALIZATION ----------
        img = cv2.imread(filepath)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        blurred = cv2.GaussianBlur(gray, (5,5), 0)

        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        kernel = np.ones((3,3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        output_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > 700:
                (x, y, w, h) = cv2.boundingRect(cnt)
                center = (x + w // 2, y + h // 2)
                radius = int(max(w, h) / 2)

                cv2.circle(output_img, center, radius, (0, 255, 255), 3)

        highlighted_path = os.path.join(UPLOAD_FOLDER, "highlighted_" + filename)
        cv2.imwrite(highlighted_path, output_img)

        predictions_collection.insert_one({
            "username": session["username"],
            "prediction": predicted_class,
            "confidence": confidence,
            "timestamp": datetime.now()
        })

        return render_template(
            "result.html",
            prediction=predicted_class,
            confidence=round(confidence * 100, 2),
            image_path=url_for("static", filename="uploads/highlighted_" + filename)
        )

    return render_template("upload_img.html")


# -------- LOGOUT --------
@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
