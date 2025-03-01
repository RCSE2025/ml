from os import environ
from dotenv import load_dotenv

load_dotenv()

WORKERS_COUNT = int(environ.get("WORKERS_COUNT", 1))
HOST = environ.get("HOST", "localhost")
