import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-please-change'
    # Use raw string for Windows path to avoid escape character issues
    AEDT_PROJECTS_BASE_DIR = r"D:\simulations\simutations\#iftx"
    AEDT_VERSION = "2025.2"
    DEBUG = True
