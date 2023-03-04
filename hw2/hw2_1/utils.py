import argparse
from pathlib import Path


def dir_path(path):
    if Path(path).is_dir():
        return path
    else:
        raise argparse.ArgumentTypeError("Must be a directory")


def is_text_file(filename):
    if filename.endswith(".txt"):
        return filename
    else:
        raise argparse.ArgumentTypeError("Must be a text file")
