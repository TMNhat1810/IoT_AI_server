from app import app
from threading import Thread
from app.service import process, get_distance

if __name__ == "__main__":
    Thread(target=process).start()
    Thread(target=get_distance).start()
    app.run(host="0.0.0.0", port=5000)
