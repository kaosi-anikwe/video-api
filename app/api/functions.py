import os
import time
import json
import imghdr
import random
import requests
from datetime import datetime
from midjourney_api import TNL
from dotenv import load_dotenv

load_dotenv()

# img2vid
from pytorch_lightning import seed_everything
from .streamlit_helpers import *


TNL_API_KEY = os.getenv("TNL_API_KEY")
TNL_IMAGE_URL = "https://api.thenextleg.io/getImage"
VERSION2SPECS = {
    "svd": {
        "T": 14,
        "H": 576,
        "W": 1024,
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd.yaml",
        "ckpt": "checkpoints/svd.safetensors",
        "options": {
            "discretization": 1,
            "cfg": 2.5,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 2,
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 25,
        },
    },
    "svd_image_decoder": {
        "T": 14,
        "H": 576,
        "W": 1024,
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd_image_decoder.yaml",
        "ckpt": "checkpoints/svd_image_decoder.safetensors",
        "options": {
            "discretization": 1,
            "cfg": 2.5,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 2,
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 25,
        },
    },
    "svd_xt": {
        "T": 25,
        "H": 576,
        "W": 1024,
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd.yaml",
        "ckpt": "checkpoints/svd_xt.safetensors",
        "options": {
            "discretization": 1,
            "cfg": 3.0,
            "min_cfg": 1.5,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 2,
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 30,
            "decoding_t": 14,
        },
    },
    "svd_xt_image_decoder": {
        "T": 25,
        "H": 576,
        "W": 1024,
        "C": 4,
        "f": 8,
        "config": "configs/inference/svd_image_decoder.yaml",
        "ckpt": "checkpoints/svd_xt_image_decoder.safetensors",
        "options": {
            "discretization": 1,
            "cfg": 3.0,
            "min_cfg": 1.5,
            "sigma_min": 0.002,
            "sigma_max": 700.0,
            "rho": 7.0,
            "guider": 2,
            "force_uc_zero_embeddings": ["cond_frames", "cond_frames_without_noise"],
            "num_steps": 30,
            "decoding_t": 14,
        },
    },
}


def is_image(file_path):
    # Use imghdr.what to determine the image type
    image_type = imghdr.what(file_path)
    # imghdr.what returns None if the file is not recognized as an image
    return image_type is not None


def generate_path(date=None):
    # Use the provided date or the current date if None
    if date is None:
        date = datetime.utcnow()
    # Format date components
    year = date.year
    month = date.month
    day = date.day
    # Format path
    path = os.path.join(f"{month:02d}-{year}", f"{day:02d}-{month:02d}")
    return path


def download_image(url):
    response = requests.get(url)
    logger.info(f"IMAGE DOWNLOAD: {response.status_code}")
    if response.status_code == 200:
        # Create a temporary file to save the image
        _, temp_filename = tempfile.mkstemp(suffix=".jpg")

        with open(temp_filename, "wb") as temp_file:
            temp_file.write(response.content)

        return temp_filename
    else:
        logger.info("Trying Midjourney safe download.")
        payload = json.dumps({
        "imgUrl": url
        })
        headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/arraybuffer',
        'Authorization': f'Bearer {TNL_API_KEY}'
        }
        response = requests.post(TNL_IMAGE_URL, headers=headers, data=payload)
        logger.info(f"IMAGE DOWNLOAD: {response.status_code}")
        if response.status_code == 200:
            # Create a temporary file to save the image
            _, temp_filename = tempfile.mkstemp(suffix=".jpg")

            with open(temp_filename, "wb") as temp_file:
                temp_file.write(response.content)

            return temp_filename
        return None


def do_img2vid(request, image: str, video_record=None):
    version = "svd_xt"
    version_dict = VERSION2SPECS[version]
    H = int(request.get("H", version_dict["H"]))
    W = int(request.get("W", version_dict["W"]))
    T = int(request.get("T", version_dict["T"]))
    C = version_dict["C"]
    F = version_dict["f"]
    options = version_dict["options"]
    state = init_st(version_dict, load_filter=True)
    model = state["model"]
    ukeys = set(get_unique_embedder_keys_from_conditioner(state["model"].conditioner))
    value_dict = init_embedder_options(request, ukeys, {}, image)

    value_dict["image_only_indicator"] = 0

    img = load_img_for_prediction(W, H, image)
    cond_aug = float(request.get("cond_aug", 0.02))
    value_dict["cond_frames_without_noise"] = img
    value_dict["cond_frames"] = img + cond_aug * torch.randn_like(img)
    value_dict["cond_aug"] = cond_aug
    seed = int(request.get("seed", 23))
    seed_everything(seed)

    options["num_frames"] = T

    sampler, num_rows, num_cols = init_sampling(request, options=options)
    num_samples = num_rows * num_cols

    decoding_t = int(request.get("decoding_t", options.get("decoding_t", T)))

    if request.get("fps"):
        saving_fps = int(request.get("fps"))
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

    results = save_video_as_grid_and_mp4(
        samples, T, video_record=video_record, fps=saving_fps
    )

    return results if results else None


def text2img(prompt: str):
    tnl = TNL(TNL_API_KEY)
    response = tnl.imagine(f"{prompt} --ar 16:9")
    logger.info(response)
    if not response.get("success"):
        return None
    message_id = response["messageId"]
    results = None
    while True:
        # check for status
        status = tnl.get_message_and_progress(message_id, 30)
        logger.info(status)
        if status["progress"] == 100:
            results = status.get("response", {})
            break
        time.sleep(1)

    if results:
        # get image url
        image_url = random.choice(results["imageUrls"])
        return image_url
    return None
