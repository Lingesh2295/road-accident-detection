import os
import cv2
import numpy as np
from flask import Flask, render_template, request,Response
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from keras.models import load_model
from keras_preprocessing.image import img_to_array
import serial

app = Flask(__name__)
app.secret_key = "7103"

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)

model.load_weights("model_weights.keras")

print("Model Loaded Successfully")

classes = ['major accident', 'minor accident', 'non Accident']



def predict_image(img_path):

    img = image.load_img(img_path, target_size=(224,224))
    img = image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = preprocess_input(img)

    prediction = model.predict(img)

    print("Prediction Probabilities:", prediction)

    class_index = np.argmax(prediction)

    result = classes[class_index]

    return result



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
    #print(result)
    if result == 'major accident' or result == 'minor accident':
        print('serial',result)
##        SerialObj = serial.Serial('COM7')
##        SerialObj.baudrate = 9600
##        SerialObj.bytesize = 8
##        SerialObj.parity   ='N'
##        SerialObj.stopbits = 1
##        SerialObj.write(b'a')
##        SerialObj.close()

    return render_template(
        "result.html",
        prediction=result,
        img_path=filepath
    )

drowsy_model = load_model("drowiness_new7.h5")

face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier("data/haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier("data/haarcascade_righteye_2splits.xml")

count = 0


def generate_frames():

    global count

    cap = cv2.VideoCapture(0)

    while True:

        success, frame = cap.read()

        if not success:
            break

        height = frame.shape[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            left_eye = left_eye_cascade.detectMultiScale(roi_gray)
            right_eye = right_eye_cascade.detectMultiScale(roi_gray)
            for (x1, y1, w1, h1) in left_eye:
                cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)
                eye1 = roi_color[y1:y1+h1, x1:x1+w1]
                eye1 = cv2.resize(eye1, (145, 145))
                eye1 = eye1.astype('float') / 255.0
                eye1 = img_to_array(eye1)
                eye1 = np.expand_dims(eye1, axis=0)
                pred1 = drowsy_model.predict(eye1)
                status1=np.argmax(pred1)
                break

            for (x2, y2, w2, h2) in right_eye:
                cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)
                eye2 = roi_color[y2:y2 + h2, x2:x2 + w2]
                eye2 = cv2.resize(eye2, (145, 145))
                eye2 = eye2.astype('float') / 255.0
                eye2 = img_to_array(eye2)
                eye2 = np.expand_dims(eye2, axis=0)
                pred2 = drowsy_model.predict(eye2)
                status2=np.argmax(pred2)
                break

            
            if status1 == 2 and status2 == 2:
                count += 1
                cv2.putText(frame, "Eyes Closed, Frame count: " + str(count), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                if count >= 8:
                    cv2.putText(frame, "Drowsiness Alert!!!", (100, height-20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
##                    SerialObj = serial.Serial('COM7')
##                    SerialObj.baudrate = 9600
##                    SerialObj.bytesize = 8
##                    SerialObj.parity   ='N'
##                    SerialObj.stopbits = 1
##                    SerialObj.write(b'a')
##                    SerialObj.close()
                    
            else:
                cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                count = 0
                alarm_on = False

       
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(),
    mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == "__main__":
    app.run(debug=False, port=7103)
