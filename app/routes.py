from app import app
from .handler import Handler
from .AI import *

@app.route('/upload', methods=['POST'])
def upload_image():
    return Handler.handle_upload()