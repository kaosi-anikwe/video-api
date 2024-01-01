# local imports
from app import logger
from app.models import Videos
from .functions import is_image, text2img, download_image

# flask imports
from flask import Blueprint, request, jsonify, send_file, json

# other imports
import os
import time
import tempfile
import traceback
from dotenv import load_dotenv

load_dotenv()

api = Blueprint("api", __name__)


@api.get("/")
def index():
    return "Hello World"


@api.get("/download/video/<video_id>")
def download_video(video_id):
    try:
        video = Videos.query.filter(Videos.uid == video_id).one_or_none()
        if not video:
            return (
                jsonify(
                    error="Not found", message="The requested video was not found."
                ),
                404,
            )
        video_path = video.video_path()
        logger.info(f"VIDEO DOWNLOAD PATH: {video_path}")
        if os.path.exists(video_path):
            return send_file(
                video_path,
                mimetype="video/mp4",
                download_name=video.video_name,
                as_attachment=True,
            )
        return (
            jsonify(error="Not found", message="The requested video no longer exists."),
            404,
        )
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify(error="Internal server error", message=str(e))


@api.get("/download/thumbnail/<thumbnail_id>")
def download_thumbnail(thumbnail_id):
    try:
        thumbnail = Videos.query.filter(Videos.uid == thumbnail_id).one_or_none()
        if not thumbnail:
            return (
                jsonify(
                    error="Not found", message="The requested thumbnail was not found."
                ),
                404,
            )
        thumbnail_path = thumbnail.thumbnail_path()
        if os.path.exists(thumbnail_path):
            return send_file(
                thumbnail_path,
                mimetype="image/jpeg",
                download_name=thumbnail.thumbnail_name,
                as_attachment=True,
            )
        return (
            jsonify(
                error="Not found", message="The requested thumbnail no longer exists."
            ),
            404,
        )
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify(error="Internal server error", message=str(e))


@api.get("/status")
def get_status():
    try:
        uid = request.args.get("id")
        if uid:
            video = Videos.query.filter(Videos.uid == uid).one_or_none()
            if not video:
                return (
                    jsonify(
                        error="Not found", message="The requested record was not found."
                    ),
                    404,
                )
            return jsonify(video.format())
        videos = Videos.query.all()
        videos = [video.format() for video in videos]
        return jsonify(tasks=videos)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify(error="Internal server error", message=str(e))


@api.post("/img2vid")
def img2vid():
    data = request.get_json()
    try:
        logger.info(data)
        if not data.get("prompt") and not data.get("image_url"):
            return (
                jsonify(error="Invalid request", message="Image and prompt not found"),
                400,
            )

        if not data.get("image_url"):
            image = download_image(text2img(data["prompt"]))
        else:
            image = download_image(data["image_url"])

        if not image:
            return (
                jsonify(error="Internal server error", message="Error retrieving image"),
                500,
            )           
        if not is_image(image):
            return (
                jsonify(error="Invalid request", message="Invalid image file"),
                400,
            )

        # save temp config
        with tempfile.NamedTemporaryFile("w+", delete=False) as tmp_conf:
            json.dump(data, tmp_conf)
            tmp_conf_name = tmp_conf.name

        new_video = Videos("svd_xt", image, tmp_conf_name)
        new_video.insert()
        return jsonify(new_video.format())
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify(error="Internal server error", message=str(e))
