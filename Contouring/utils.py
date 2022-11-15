import os
import datetime


def time():
    return datetime.datetime.now()
    

def make_folder(out_path, specification):
    composed = f"{out_path}/{specification}"
    exists = os.path.exists(composed)
    if not exists:
        os.mkdir(composed)

    return composed
