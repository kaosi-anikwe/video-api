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
from app.api.functions import do_img2vid

load_dotenv()

# initialize firebase app
SERVICE_CERT = json.loads(os.getenv("SERVICE_CERT"))
STORAGE_BUCKET = os.getenv("STORAGE_BUCKET")
cred_obj = firebase_admin.credentials.Certificate(SERVICE_CERT)
firebase_admin.initialize_app(cred_obj, {"storageBucket": STORAGE_BUCKET})
db = firestore.client()
ref = db.collection("videoList").document()


def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        # Create a temporary file to save the image
        _, temp_filename = tempfile.mkstemp(suffix=".jpg")

        with open(temp_filename, "wb") as temp_file:
            temp_file.write(response.content)

        return temp_filename
    else:
        return None


def videoRecordDict(userID):
    return {
        "addToFeed": False,
        "commentsCount": 0,
        "likes": [],
        "shares": [],
        "thumbnail": "",
        "uploaderID": userID,
        "videoCaption": "",
        "videoURL": "",
        "startTime": datetime.utcnow(),
        "endTime": None,
        "status": "processing",
    }


def handler(job):
    # get job input
    request = job["input"]
    video_record = videoRecordDict(request.get("userID", ""))
    # add data to firebase
    ref.set(video_record)
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
