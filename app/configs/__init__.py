import os
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

UPLOAD_FOLDER = os.path.join(os.getcwd(), "app", "static", "uploads")

CONFIDENT_THRESHOLD = 0.6

RECOGNIZE_CONFIDENT = 0.99999999

STREAM_URL = "http://192.168.133.17:81/stream"

CENTER_SERVER_URL = "http://192.168.133.112:8000"

ESP_URL = "http://192.168.133.233"

OFFSET = 24

RF_DISTANCE_THRESHOLD = 40
