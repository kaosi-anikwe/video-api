import os
from dotenv import load_dotenv

load_dotenv()

# define base directory of app
basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    # key for CSF
    SECRET_KEY = os.environ.get("SECRET_KEY")
    # sqlalchemy .db location (for sqlite)
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", "sqlite:///database.db")
    # sqlalchemy track modifications in sqlalchemy
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECURITY_PASSWORD_SALT = os.environ.get("SECURITY_PASSWORD_SALT")
