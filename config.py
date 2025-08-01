import os, json, sys
from PySide6.QtCore import QStandardPaths

# Определяем базовый каталог приложения
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(__file__)

# Дефолтные папки внутри BASE_DIR
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")
DOWNLOADS_DIR = QStandardPaths.writableLocation(QStandardPaths.DownloadLocation)
RESULTS_DIR = os.path.join(BASE_DIR, "results") 

# Путь к _internal/config.json
config_path = os.path.join(BASE_DIR, "config.json")

# Чтение и загрузка
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)

# Используем переменные
CLASS_NAMES = config.get("class_names", [])