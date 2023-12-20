# local imports
from app import logger
from app.models import Videos
from .fuctions import is_image, VERSION2SPECS
from .streamlit_helpers import *

# flask imports
from flask import Blueprint, request, jsonify, send_file

# other imports
import os
import time
import tempfile
import traceback
from dotenv import load_dotenv
from pytorch_lightning import seed_everything

load_dotenv()

api = Blueprint("api", __name__)

SAVE_PATH = os.getenv("VIDEO_DIR")


@api.get("/")
def index():
    return "Hello World"


@api.get("/download/video/<video_id>")
def download_video(video_id):
    video = Videos.query.filter(Videos.uid == video_id).one_or_none()
    if not video:
        return (
            jsonify(error="Not found", message="The requested video was not found."),
            404,
        )
    video_path = video.video_path()
    logger.info(f"VIDEO DOWNLOAD PATH: {video_path}")
    if os.path.exists(video_path):
        return send_file(
            video_path, mimetype="video/mp4", download_name=video.video_name
        )
    return (
        jsonify(error="Not found", message="The requested video no longer exists."),
        404,
    )


@api.get("/download/thumbnail/<thumbnail_id>")
def download_thumbnail(thumbnail_id):
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
        )
    return (
        jsonify(error="Not found", message="The requested thumbnail no longer exists."),
        404,
    )


@api.post("/img2vid")
def img2vid():
    try:
        start_time = time.time()
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
                return (
                    jsonify(error="Invalid request", message="Invalid image file"),
                    400,
                )
        finally:
            if os.path.exists(tmp_file_name):
                os.remove(tmp_file_name)

        version = "svd_xt"
        version_dict = VERSION2SPECS[version]
        H = int(request.form.get("H", version_dict["H"]))
        W = int(request.form.get("W", version_dict["W"]))
        T = int(request.form.get("T", version_dict["T"]))
        C = version_dict["C"]
        F = version_dict["f"]
        options = version_dict["options"]
        state = init_st(version_dict, load_filter=True)
        model = state["model"]
        ukeys = set(
            get_unique_embedder_keys_from_conditioner(state["model"].conditioner)
        )
        value_dict = init_embedder_options(ukeys, {}, image)

        value_dict["image_only_indicator"] = 0

        img = load_img_for_prediction(W, H, image)
        cond_aug = float(request.form.get("cond_aug", 0.02))
        value_dict["cond_frames_without_noise"] = img
        value_dict["cond_frames"] = img + cond_aug * torch.randn_like(img)
        value_dict["cond_aug"] = cond_aug
        seed = int(request.form.get("seed", 23))
        seed_everything(seed)

        save_path = SAVE_PATH
        options["num_frames"] = T

        sampler, num_rows, num_cols = init_sampling(options=options)
        num_samples = num_rows * num_cols

        decoding_t = int(request.form.get("decoding_t", options.get("decoding_t", T)))

        if request.form.get("fps"):
            saving_fps = int(request.form.get("fps"))
        else:
            saving_fps = value_dict["fps"]

        out = do_sample(
            model,
            sampler,
            value_dict,
            num_samples,
            H,
            W,
            C,
            F,
            T=T,
            batch2model_input=["num_video_frames", "image_only_indicator"],
            force_uc_zero_embeddings=options.get("force_uc_zero_embeddings", None),
            force_cond_zero_embeddings=options.get("force_cond_zero_embeddings", None),
            return_latents=False,
            decoding_t=decoding_t,
        )

        if isinstance(out, (tuple, list)):
            samples, samples_z = out
        else:
            samples = out
            samples_z = None

        new_video = Videos(version)
        save_video_as_grid_and_mp4(
            samples, save_path, T, video_record=new_video, fps=saving_fps
        )
        new_video.insert()

        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(elapsed_time, 60)

        formatted_time = "{:02}:{:02}".format(int(minutes), int(seconds))

        response = {
            "id": new_video.uid,
            "video_url": new_video.video_url(),
            "thumbnail_url": new_video.thumbnail_url(),
            "time": formatted_time,
        }

        return jsonify(response)

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify(error="Internal server error", message=str(e))
