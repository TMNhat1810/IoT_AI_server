from app import app
from .handler import Handler
from .AI import *
from flask import Response, stream_with_context
from .service import stream


@app.route("/upload", methods=["POST"])
def upload_image():
    return Handler.handle_upload()


@app.route("/stream")
def video_feed():
    return Response(
        stream_with_context(stream()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/distance", methods=["POST"])
def update_distance():
    return Handler.handle_distance()
