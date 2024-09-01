from flask import Flask, Response, render_template, jsonify, request
import cv2
import numpy as np

app = Flask(__name__)


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


cap = cv2.VideoCapture(0)


color_recognized = (0, 255, 0)  
color_unrecognized = (0, 0, 255)  
font = cv2.FONT_HERSHEY_SIMPLEX
confidence_threshold = 100  

def preprocess_face(face):
    """Preprocess the face image to improve recognition."""
    face = cv2.equalizeHist(face)  
    face = cv2.GaussianBlur(face, (3, 3), 0)  
    face = cv2.resize(face, (200, 200))  
    return face

def align_face(face):
    """Align the face image to improve recognition."""
    return face

def gen_frames():
    """Generate frames from the webcam."""
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            preprocessed_face = preprocess_face(face)
            aligned_face = align_face(preprocessed_face)
            label, confidence = recognizer.predict(aligned_face)

           
            eyes = eye_cascade.detectMultiScale(face)

            
            if confidence < confidence_threshold and len(eyes) >= 2:
                color = color_recognized
                text = 'Welcome Sir'
            else:
                color = color_unrecognized
                text = 'Cannot enter'

            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), font, 0.9, color, 2, cv2.LINE_AA)

        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status', methods=['GET'])
def status():
    
    return jsonify({
        'status': 'Authentication in progress',
        'progress': '50%'
    })

@app.route('/start_auth', methods=['POST'])
def start_auth():
    
    return jsonify({'status': 'Authentication started'})

@app.route('/stop_auth', methods=['POST'])
def stop_auth():
    
    return jsonify({'status': 'Authentication stopped'})

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        cap.release()
        cv2.destroyAllWindows()