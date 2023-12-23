import os
import json
import time
import traceback
from app import create_app, logger, db
from app.models import Videos
from app.api.functions import do_img2vid


if __name__ == "__main__":
    while True:
        app = create_app()
        with app.app_context():
            db.create_all()
            logger.info("Checking for queued tasks.")
            videos = Videos.query.filter(Videos.status == "queued").all()
            for video in videos:
                logger.info(f"Processing video with id: {video.uid}")
                video.update_status("processing")
                with open(video.tmp_conf_file) as conf_file:
                    request = json.load(conf_file)
                image = video.tmp_image_path
                try:
                    do_img2vid(request, image, video)
                    logger.info(f"Done processing video with id: {video.uid}")
                except:
                    logger.info(f"Error processing video with id: {video.uid}")
                    logger.error(traceback.format_exc())
                    video.update_status("error")
                finally:
                    if os.path.exists(video.tmp_image_path):
                        os.remove(video.tmp_image_path)
                    if os.path.exists(video.tmp_conf_file):
                        os.remove(video.tmp_conf_file)
        time.sleep(10)
