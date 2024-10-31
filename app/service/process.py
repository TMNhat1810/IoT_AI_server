from ..AI import detect
from imutils.video import VideoStream
from ..configs import STREAM_URL, CENTER_SERVER_URL, SERVO_URL
from ..configs.detect import *
from .. import globals
import time
import requests
import cv2
import threading


def log_fake(buffer):
    try:
        files = {"file": ("capture.jpg", buffer, "image/jpeg")}
        requests.post(CENTER_SERVER_URL + "/image/fake", files=files)
    except:
        pass
    time.sleep(10)
    globals._stop_detect = False


def log_real(buffer, label):
    try:
        files = {"file": ("capture.jpg", buffer, "image/jpeg")}
        requests.post(
            SERVO_URL,
            json={"isServo": True},
        )
        requests.post(
            CENTER_SERVER_URL + "/image/real", files=files, data={"title": label}
        )
    except:
        pass
    time.sleep(10)
    globals._stop_detect = False


def process():
    while True:
        try:
            vs = VideoStream(STREAM_URL).start()
            time.sleep(1.0)
            break
        except:
            vs.stop()
            time.sleep(1.0)
            pass
    rc = 0
    fc = 0
    nc = 0

    n_fake = 0
    n_real = 0

    while True:
        try:
            frame = vs.read()
            rf_label, rc_label = detect(frame)
            _, buffer = cv2.imencode(".jpg", frame)
            globals._frame = buffer.tobytes()

            if globals._stop_detect == True:
                continue

            if not rf_label:
                nc += 1
            elif rf_label == "fake":
                fc += 1
                nc = 0
            else:
                rc += 1
                nc = 0

            if nc == N_NONE_RESET:
                rc = 0
                fc = 0
                nc = 0
                n_fake = 0
            if fc == N_FAKE_FRAME:
                print("fake")
                print(rc_label)
                rc = 0
                fc = 0
                nc = 0
                n_fake += 1
            if rc == N_REAL_FRAME:
                print("real")
                print(rc_label)
                rc = 0
                fc = 0
                nc = 0
                n_fake = 0
                n_real += 1

            if n_fake == 3:
                threading.Thread(target=log_fake, args=(buffer,)).start()
                n_fake = 0
                globals._stop_detect = True

            if n_real == 3:
                threading.Thread(
                    target=log_real,
                    args=(
                        buffer,
                        rc_label,
                    ),
                ).start()
                n_real = 0
                globals._stop_detect = True
        except:
            pass
    process()
