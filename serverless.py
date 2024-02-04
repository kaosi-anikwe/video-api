import os
import json
import runpod
import tempfile
import requests
import traceback
import firebase_admin
from datetime import datetime
from dotenv import load_dotenv
from firebase_admin import firestore

# local imports
from app.api.functions import do_img2vid, text2img, download_image

load_dotenv()

# initialize firebase app
SERVICE_CERT = json.loads(os.getenv("SERVICE_CERT"))
STORAGE_BUCKET = os.getenv("STORAGE_BUCKET")
cred_obj = firebase_admin.credentials.Certificate(SERVICE_CERT)
firebase_admin.initialize_app(cred_obj, {"storageBucket": STORAGE_BUCKET})
db = firestore.client()
ref = db.collection("videosList").document()


def videoRecordDict(userID, prompt):
    return {
        "addToFeed": False,
        "commentsCount": 0,
        "likes": [],
        "shares": [],
        "thumbnail": "",
        "uploaderId": userID,
        "videoCaption": prompt,
        "videoUrl": "",
        "createdAt": None,
    }


def handler(job):
    # get job input
    request = job["input"]
    video_record = videoRecordDict(request.get("userID", ""), request.get("prompt", ""))
    # add data to firebase
    ref.set(video_record)
    if not request.get("image_url"):
        image = download_image(text2img(request["prompt"]))
    else:
        image = download_image(request["image_url"])
    if not image:
        return {"error": "Failed to download image"}
    try:
        result = do_img2vid(request, image)
        ref.update(result)
        return result
    except Exception as e:
        ref.update({"status": "error", "errorMessage": traceback.format_exc()})
        return {"error": str(e)}
    finally:
        if os.path.exists(image):
            os.remove(image)


runpod.serverless.start({"handler": handler})
