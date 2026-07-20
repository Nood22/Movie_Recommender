import os
import pandas as pd

# مسیر درست به پوشه data داخل Code4Neda
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE4NEDA_PATH = os.path.join(BASE_DIR, "Code4Neda")
DATA_PATH = os.path.join(CODE4NEDA_PATH, "data")

def map_id_to_title(data_name="ml-1m"):
    movies_file = os.path.join(DATA_PATH, data_name, "movies.dat")
    mapping = {}
    with open(movies_file, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) == 3:
                mid, title, genres = parts
                mapping[int(mid)] = title
    return mapping

def map_id_to_genre(data_name="ml-1m"):
    movies_file = os.path.join(DATA_PATH, data_name, "movies.dat")
    mapping = {}
    with open(movies_file, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) == 3:
                mid, title, genres = parts
                mapping[int(mid)] = genres.split("|")
    return mapping
