from flask import request, jsonify
from .. import globals


def handle_distance():
    try:
        globals._distance = request.json["distance"]
        return jsonify({"success": True}), 200
    except:
        return jsonify({"success": False}), 500
