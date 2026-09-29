from flask import Flask, render_template, Response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import winsound

app = Flask(__name__)
CORS(app)

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

MAX_CAPACITY = 10
current_count = 0

def generate_frames():
    global current_count
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        count = 0

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if model.names[cls] == "person":
                    count += 1
                    
        current_count = count
        cv2.putText(frame, f"People Count: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if count > MAX_CAPACITY:
            cv2.putText(frame, "OVER CAPACITY!", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            winsound.Beep(1500, 800)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/count')
def count():
    return {
        "people_count": current_count,
        "capacity": MAX_CAPACITY,
        "status": "OVER CAPACITY" if current_count > MAX_CAPACITY else "NORMAL"
    }

@app.route('/video')
@app.route('/video_feed')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)