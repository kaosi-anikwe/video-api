import os
import logging
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

# flask imports
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# local imports
from config import Config

load_dotenv()

db = SQLAlchemy()

os.makedirs("log", exist_ok=True)

# configure logger
log_filename = os.path.join("log", "run.log")
log_max_size = 1 * 1024 * 1024  # 1 MB
# Create a logger
logger = logging.getLogger("svd-api")
logger.setLevel(logging.INFO)
# Create a file handler with log rotation
handler = RotatingFileHandler(log_filename, maxBytes=log_max_size, backupCount=5)
# Create a formatter
formatter = logging.Formatter("%(asctime)s [%(levelname)s] - %(message)s")
handler.setFormatter(formatter)
# Add the handler to the logger
logger.addHandler(handler)


def create_app(config=Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config)
    db.init_app(app)

    from .api.routes import api

    app.register_blueprint(api)

    return app
