import os
import imghdr
from datetime import datetime

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
