from keras.models import load_model
from keras.layers import TFSMLayer
import pickle
import os


os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"

print("[INFO] loading liveness detector...")
# model = load_model(
#     os.path.sep.join(
#         [os.getcwd(), "app", "AI", "real_fake", "model", "_liveness.model"]
#     )
# )
model = TFSMLayer(
    os.path.sep.join(
        [os.getcwd(), "app", "AI", "real_fake", "model", "liveness.model"]
    ),
    call_endpoint="serving_default",
)
label_path = os.path.sep.join(
    [os.getcwd(), "app", "AI", "real_fake", "model", "le.pickle"]
)
try:
    le = pickle.loads(open(label_path, "rb").read())
except:
    le = pickle.loads(open(label_path, "rb").read().replace(b"label", b"_label"))
