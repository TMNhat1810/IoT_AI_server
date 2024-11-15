import pickle
import os
import numpy as np
from keras_facenet import FaceNet
from tensorflow.keras.models import load_model

embedder = FaceNet()

faces = ["DucHuy", "Nhat", "Quoc", "Toan"]

recognizer = load_model(
    os.path.join(
        os.getcwd(),
        "app",
        "AI",
        "face_recognizer",
        "model",
        "FaceNetCNN.h5",
    )
)


def get_embedding(face_img):
    face_img = face_img.astype("float32")  # 3D(160x160x3)
    face_img = np.expand_dims(face_img, axis=0)
    # 4D (Nonex160x160x3)
    yhat = embedder.embeddings(face_img)
    return yhat[0]  # 512D image (1x1x512)
