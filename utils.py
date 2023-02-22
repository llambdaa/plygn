import os
import datetime

from pathlib import Path

def time():
    return datetime.datetime.now()
    

def make_folder(out_path, specification):
    composed = f"{out_path}/{specification}"
    exists = os.path.exists(composed)
    if not exists:
        os.mkdir(composed)

    return composed


def truncate_path(target, length):
    path = Path(target)
    parts = path.parts
    if length >= len(parts):
        return path
    
    return Path(".../", *parts[-length:])
