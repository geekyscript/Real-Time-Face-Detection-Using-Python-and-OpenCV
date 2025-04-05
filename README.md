# Real-Time Face and Object Detection Using Python and OpenCV

Computer vision has made massive leaps in recent years, enabling everyday applications like face detection, object tracking, and even autonomous vehicles. In this tutorial, we'll build a **real-time face and object detection system** using Python and OpenCV. We'll cover face detection with Haar Cascades and animal detection using MobileNet SSD.

---

![Face Detection Example](../images/img3.png)
*Example of a face detected via webcam using Haar Cascade.*

---

## 🧰 Prerequisites

Before we dive into the code, ensure the following libraries and files are available:

### Install Required Packages:
```bash
pip install opencv-python numpy
```

### Download Pre-trained Models:
For object detection, download the following and place them in your working directory:
- `MobileNetSSD_deploy.caffemodel`
- `MobileNetSSD_deploy.prototxt`

You can find these [here](https://github.com/chuanqi305/MobileNet-SSD).

---

## 👦 Face Detection Using Webcam (Haar Cascade)

```python
import cv2

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```
