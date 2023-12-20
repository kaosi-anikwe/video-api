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

# Logging configuration
log_filename = os.path.join("log", "run.log")
log_max_size = 1 * 1024 * 1024  # 1 MB
# configure logger
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("svd-api")
# Create a file handler with log rotation
handler = RotatingFileHandler(log_filename, maxBytes=log_max_size, backupCount=5)
# Add the handler to the logger
logger.addHandler(handler)


def create_app(config=Config) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config)
    db.init_app(app)

    from .api.routes import api

    app.register_blueprint(api)

    return app
