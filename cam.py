from flask import Flask, render_template, Response
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load your Keras model (edit path as needed)
model = tf.keras.models.load_model("gesture_model.h5")

camera = cv2.VideoCapture(0)

def preprocess(frame):
    frame_resized = cv2.resize(frame, (64, 64))
    frame_normalized = frame_resized / 255.0
    return np.expand_dims(frame_normalized, axis=0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Predict gesture
            input_data = preprocess(frame)
            prediction = model.predict(input_data)
            gesture = np.argmax(prediction)

            # Draw prediction
            cv2.putText(frame, f"Prediction: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Encode to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Stream to HTML
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # Loads HTML page

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
