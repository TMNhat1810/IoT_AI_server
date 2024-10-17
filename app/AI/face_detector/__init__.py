import cv2
import os

print("[INFO] loading face detector...")

protoPath = os.path.sep.join([os.getcwd(),"app", "AI", "face_detector", 
                              "model", "deploy.prototxt"])
modelPath = os.path.sep.join([os.getcwd(),"app", "AI", "face_detector", 
                              "model", "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
