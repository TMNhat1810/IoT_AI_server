from app import app
from threading import Thread
from app.service import detect

if __name__ == "__main__":
    Thread(target=detect).start()
    app.run(host="0.0.0.0", port=5000)
