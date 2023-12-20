# local imports
from . import db
from .api.functions import generate_path

# flask imports
from flask import url_for

# other imports
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

VIDEO_DIR = os.getenv("VIDEO_DIR")
THUMBNAIL_DIR = os.getenv("THUMBNAIL_DIR")

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(THUMBNAIL_DIR, exist_ok=True)


# timestamp to be inherited by other class models
class TimestampMixin(object):
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)

    def format_date(self):
        self.created_at = self.created_at.strftime("%d %B, %Y %I:%M")

    def format_time(self):
        try:
            self.datetime = self.datetime.strftime("%d %B, %Y %I:%M")
        except:
            pass


# db helper functions
class DatabaseHelperMixin(object):
    def update(self):
        db.session.commit()

    def insert(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()


class Videos(db.Model, TimestampMixin, DatabaseHelperMixin):
    __tablename__ = "video"

    id = db.Column(db.Integer, primary_key=True)
    uid = db.Column(db.String(200), unique=True, nullable=False)
    version = db.Column(db.String(20), nullable=False)
    video_name = db.Column(db.String(200), unique=True)
    thumbnail_name = db.Column(db.String(200), unique=True)
    status = db.Column(db.String(10))
    tmp_image_path = db.Column(db.String(500))
    tmp_conf_file = db.Column(db.String(500))
    start_time = db.Column(db.DateTime)
    end_time = db.Column(db.DateTime)

    def __init__(self, version: str, tmp_image_path: str, tmp_conf_file) -> None:
        self.uid = uuid.uuid4().hex
        self.version = version
        self.video_name = f"{self.uid}.mp4"
        self.thumbnail_name = f"{self.uid}.png"
        self.status = "queued"
        self.tmp_image_path = tmp_image_path
        self.tmp_conf_file = tmp_conf_file

    def video_path(self) -> str:
        video_dir = os.path.join(VIDEO_DIR, generate_path(self.end_time))
        os.makedirs(video_dir, exist_ok=True)
        return os.path.join(video_dir, self.video_name)

    def thumbnail_path(self) -> str:
        thumbnail_dir = os.path.join(THUMBNAIL_DIR, generate_path(self.end_time))
        os.makedirs(thumbnail_dir, exist_ok=True)
        return os.path.join(thumbnail_dir, self.thumbnail_name)

    def video_url(self) -> str:
        return url_for("api.download_video", video_id=self.uid)

    def thumbnail_url(self) -> str:
        return url_for("api.download_thumbnail", thumbnail_id=self.uid)

    def update_status(self, status) -> None:
        if status == "processing":
            self.start_time = datetime.utcnow()
        if status == "completed":
            self.end_time = datetime.utcnow()
        self.status = status
        return self.update()

    def duration(self):
        return {
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S")
            if self.start_time
            else None,
            "end_time": self.end_time.strftime("%Y-%m-%d %H:%M:%S")
            if self.end_time
            else None,
        }

    def format(self):
        return {
            "id": self.uid,
            "status": self.status,
            "start_time": self.duration()["start_time"],
            "end_time": self.duration()["end_time"],
            "video_url": self.video_url() if self.status == "completed" else None,
            "thumbnail_url": self.thumbnail_url()
            if self.status == "completed"
            else None,
        }
