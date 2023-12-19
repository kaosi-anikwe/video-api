# flask imports
from flask import Blueprint, request, jsonify, send_file

# local imports
from app import logger
from .fuctions import is_image
from app.models import Videos

# other imports
import os
import tempfile

api = Blueprint("api", __name__)

@api.get("/")
def index():
    return "Hello World"

@api.get("/download/video/<video_id>")
def download_video(video_id):
    video = Videos.query.filter(Videos.uid == video_id).one_or_none()
    if not video:
        return jsonify(error="Not found", message="The requested video was not found."), 404
    video_path = video.video_path()
    logger.info(f"VIDEO DOWNLOAD PATH: {video_path}")
    if os.path.exists(video_path):
        return send_file(video_path, mimetype="video/mp4", download_name=video.video_name)
    return jsonify(error="Not found", message="The requested video no longer exists."), 404


@api.get("/download/thumbnail/<thumbnail_id>")
def download_thumbnail(thumbnail_id):
    thumbnail = Videos.query.filter(Videos.uid == thumbnail_id).one_or_none()
    if not thumbnail:
        return jsonify(error="Not found", message="The requested thumbnail was not found."), 404
    thumbnail_path = thumbnail.thumbnail_path()
    if os.path.exists(thumbnail_path):
        return send_file(thumbnail_path, mimetype="image/jpeg", download_name=thumbnail.thumbnail_name)
    return jsonify(error="Not found", message="The requested thumbnail no longer exists."), 404


# @api.get("/add")
# def add_sample():
#     video = Videos()
#     video.insert()
#     return jsonify(name=video.video_name, id=video.uid)

@api.post("/img2vid")
def img2vid():
    logger.info(request.files)
    if not request.files.get("file"):
        return jsonify(error="Invalid request", message="Image file not found"), 400
    image = request.files.get("file")
    try:
        with tempfile.NamedTemporaryFile("wb", delete=False) as tmp_image:
            image.stream.seek(0)
            tmp_image.write(image.stream.read())
            tmp_file_name = tmp_image.name
        if not is_image(tmp_file_name):
            return jsonify(error="Invalid request", message="Invalid image file"), 400
    finally:
        if os.path.exists(tmp_file_name):
            os.remove(tmp_file_name)
    return image.filename

