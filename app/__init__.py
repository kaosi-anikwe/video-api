import os
import logging
from dotenv import load_dotenv

# flask imports
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# local imports
from config import Config

load_dotenv()

db = SQLAlchemy()

os.makedirs("log", exist_ok=True)

# configure logger
logging.basicConfig(
    filename=os.path.join("log", "run.log"),
    level=logging.INFO,
    format="%(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("svd-api")


def create_app(config=Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config)
    db.init_app(app)

    from .api.routes import api

    app.register_blueprint(api)

    return app
