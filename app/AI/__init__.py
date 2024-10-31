import numpy as np
import cv2
from keras_preprocessing.image import img_to_array
from .real_fake import model, le
from .face_detector import net
from .face_recognizer import recognizer, get_embedding, faces
from ..configs import CONFIDENT_THRESHOLD, RECOGNIZE_CONFIDENT


def detect(frame: np.ndarray):
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

            rf = model(face)["activation_5"].numpy()[0]
            face_emb = get_embedding(
                cv2.resize(
                    frame[startY:endY, startX:endX],
                    (160, 160),
                )
            )
            recog = recognizer.predict(np.array(face_emb).reshape(1, 512), verbose=0)[0]
            recog = np.array(recog)

            rc_label = "unknown"
            if np.max(recog) > RECOGNIZE_CONFIDENT:
                rc_label = faces[np.argmax(recog)]

            j = np.argmax(rf)
            rf_label = le.classes_[j]

            if j == 0:
                cv2.putText(
                    frame,
                    rf_label,
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
                    rf_label + " " + rc_label,
                    (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            return rf_label, rc_label

    return None, None
