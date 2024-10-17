from flask import request, jsonify
from PIL import Image
import numpy as np

from ..AI import detect_real_fake

def handle_upload():
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['image']
    
    try:
        image = Image.open(file.stream).convert('RGB')

        detect_real_fake(np.array(image))

        return jsonify({
            "message": f"Image {file.filename} processed successfully",
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

# nhận request, thay vì lưu sẽ đưa vào model, TBD