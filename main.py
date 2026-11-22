import os
import cv2
import numpy as np
from flask import Flask, render_template, request, Response
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import serial

app = Flask(__name__)
app.secret_key = "7103"

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def send_serial_data(data=b'a', port='COM13', baudrate=9600):
    try:
        ser = serial.Serial(port, baudrate, bytesize=8, parity='N', stopbits=1, timeout=1)
        ser.write(data)
        ser.close()
        print(f"[SERIAL SUCCESS] Sent trigger signal '{data.decode()}' to hardware on {port}")
    except Exception as e:
        print(f"[SERIAL ERROR] Failed to send hardware signal: {e}")


# Load Model
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)
model.load_weights("model_weights.keras")
print("Accident Model Loaded Successfully")

classes = ['major accident', 'minor accident', 'non Accident']


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    return classes[class_index]


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file uploaded"

    file = request.files['image']
    if file.filename == '':
        return "No selected file"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    result = predict_image(filepath)

    if result in ['major accident', 'minor accident']:
        print('Uploaded Image Detection Triggered Hardware:', result)
        send_serial_data(b'a')

    return render_template("result.html", prediction=result, img_path=filepath)


def generate_frames():
    cap = cv2.VideoCapture(0)
    frame_count = 0
    last_detected_status = "Scanning..."

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_count += 1

        # Process every 5th frame to maintain video fluidness and performance
        if frame_count % 5 == 0:
            resized_frame = cv2.resize(frame, (224, 224))
            img_array = image.img_to_array(resized_frame)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            prediction = model.predict(img_array, verbose=0)
            class_index = np.argmax(prediction)
            result = classes[class_index]

            if result in ['major accident', 'minor accident']:
                last_detected_status = f"ACCIDENT DETECTED: {result.upper()}"
                print(f"[LIVE STREAM DETECTED] {result} - Triggering Serial Data")
                send_serial_data(b'a')
            else:
                last_detected_status = "Status: Normal (Non-Accident)"

        # Draw overlay on video feed
        color = (0, 0, 255) if "ACCIDENT" in last_detected_status else (0, 255, 0)
        cv2.putText(
            frame,
            last_detected_status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":
    app.run(debug=False, port=7103)
