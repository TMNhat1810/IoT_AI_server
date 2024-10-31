import os
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

UPLOAD_FOLDER = os.path.join(os.getcwd(), "app", "static", "uploads")

CONFIDENT_THRESHOLD = 0.6

RECOGNIZE_CONFIDENT = 0.85

STREAM_URL = "http://192.168.104.17:81/stream"

CENTER_SERVER_URL = "http://192.168.104.112:8000"

SERVO_URL = "http://192.168.104.215"
