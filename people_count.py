from ultralytics import YOLO
import cv2
import winsound

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

MAX_CAPACITY = 10

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == "person":
                count += 1

    cv2.putText(frame, f"People Count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if count > MAX_CAPACITY:
        cv2.putText(frame, "OVER CAPACITY!", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        winsound.Beep(1500, 800)

    cv2.imshow("Lift Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()