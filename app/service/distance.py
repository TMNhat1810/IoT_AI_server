import requests
import time
from .. import globals
from ..configs import ESP_URL


def get_distance():
    while True:
        try:
            response = requests.get(ESP_URL + "/distance")
            data = response.json()
            globals._distance = data["distance"]
            print(globals._distance)
            time.sleep(0.2)
        except:
            pass
