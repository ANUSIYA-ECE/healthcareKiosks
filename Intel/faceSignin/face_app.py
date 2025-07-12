from flask import Flask, render_template, request
import os
import csv
import face_recognition
from PIL import Image
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)
UPLOAD_FOLDER = 'dataset'
CSV_FILE = os.path.join(UPLOAD_FOLDER, 'users.csv')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        pass  # create empty file

@app.route('/')
def home():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    image = request.files['image']
    img_path = os.path.join(UPLOAD_FOLDER, f"{name}.jpg")

     # Force-save image as RGB
    pil_image = Image.open(image).convert("RGB")
    pil_image.save(img_path)


    with open(CSV_FILE, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([name, img_path])

    return render_template("login.html", message="✅ Registration successful!")
@app.route('/login', methods=['POST'])
def login():
    try:
        image_data = request.form['image_data']
        if not image_data:
            return render_template("login.html", message="❌ No image captured.")

        # Decode base64 image
        header, encoded = image_data.split(",", 1)
        decoded = base64.b64decode(encoded)
        image = Image.open(BytesIO(decoded)).convert("RGB")
        image_np = np.array(image)

        live_encodings = face_recognition.face_encodings(image_np)
        if not live_encodings:
            return render_template("login.html", message="❌ No face detected.")

        live_encoding = live_encodings[0]

        with open(CSV_FILE, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for name, path in reader:
                if not os.path.exists(path):
                    continue

                known_image = face_recognition.load_image_file(path)
                known_encodings = face_recognition.face_encodings(known_image)
                if not known_encodings:
                    continue

                known_encoding = known_encodings[0]
                distance = face_recognition.face_distance([known_encoding], live_encoding)[0]

                if distance < 0.5:
                    return render_template("dashboard.html", name=name)

        return render_template("login.html", message="❌ Face not recognized.")
    except Exception as e:
        print("Error:", e)
        return render_template("login.html", message="❌ Error processing image. Ensure camera works and image is clear.")
if __name__ == '__main__':
    app.run(debug=True, port=5002)
