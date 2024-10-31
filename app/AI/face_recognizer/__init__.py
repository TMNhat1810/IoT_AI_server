import pickle
import os
import numpy as np
from keras_facenet import FaceNet

embedder = FaceNet()

faces = ["DucHuy", "Nhat", "NhatHuy", "Quoc", "Toan", "unknown"]

recognizer = pickle.load(
    open(
        os.path.join(
            os.getcwd(),
            "app",
            "AI",
            "face_recognizer",
            "model",
            "svm_model_160x160.pkl",
        ),
        "rb",
    )
)


def get_embedding(face_img):
    face_img = face_img.astype("float32")  # 3D(160x160x3)
    face_img = np.expand_dims(face_img, axis=0)
    # 4D (Nonex160x160x3)
    yhat = embedder.embeddings(face_img)
    return yhat[0]  # 512D image (1x1x512)
