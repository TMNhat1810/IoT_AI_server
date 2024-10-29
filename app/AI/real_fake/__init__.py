from imutils.video import VideoStream
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import pickle
import cv2
import os
import time

from ..face_detector import net
from ...configs import CONFIDENT_THRESHOLD, UPLOAD_FOLDER

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"

print("[INFO] loading liveness detector...")
model = load_model(
    os.path.sep.join([os.getcwd(), "app", "AI", "real_fake", "model", "liveness.model"])
)
label_path = os.path.sep.join(
    [os.getcwd(), "app", "AI", "real_fake", "model", "le.pickle"]
)
try:
    le = pickle.loads(open(label_path, "rb").read())
except:
    le = pickle.loads(open(label_path, "rb").read().replace(b"label", b"_label"))


def detect_real_fake(frame: np.ndarray):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence > CONFIDENT_THRESHOLD:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w, endX)
            endY = min(h, endY)

            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (32, 32))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            preds = model.predict(face, verbose=0)[0]

            j = np.argmax(preds)
            label = le.classes_[j]

            if j == 0:
                cv2.putText(
                    frame,
                    label,
                    (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            else:
                cv2.putText(
                    frame,
                    label,
                    (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            return label
