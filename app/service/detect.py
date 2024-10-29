from ..AI import detect_real_fake
from imutils.video import VideoStream
from ..configs import STREAM_URL, CENTER_SERVER_URL
from ..configs.detect import *
from .. import globals
import time
import requests
import cv2
import threading


def log_fake(buffer):
    try:
        files = {"file": ("capture.jpg", buffer, "image/jpeg")}
        requests.post(CENTER_SERVER_URL + "/fake", files=files)
    except:
        pass
    time.sleep(5)
    globals._stop_detect = False


def log_real(buffer, label):
    try:
        files = {"file": ("capture.jpg", buffer, "image/jpeg")}
        requests.post(CENTER_SERVER_URL + "/real", files=files, json={"label": label})
    except:
        pass
    time.sleep(5)
    globals._stop_detect = False


def detect():
    try:
        vs = VideoStream(0).start()
        time.sleep(1.0)
    except:
        return
    rc = 0
    fc = 0
    nc = 0

    n_fake = 0
    n_real = 0

    while True:
        try:
            frame = vs.read()
            label = detect_real_fake(frame)
            _, buffer = cv2.imencode(".jpg", frame)
            globals._frame = buffer.tobytes()

            if globals._stop_detect == True:
                time.sleep(0.1)
                continue

            if not label:
                nc += 1
            elif label == "fake":
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
                rc = 0
                fc = 0
                nc = 0
                n_fake += 1
            if rc == N_REAL_FRAME:
                print("real")
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
                        label,
                    ),
                ).start()
                n_real = 0
                globals._stop_detect = True
        except:
            pass
