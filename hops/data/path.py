import pathlib

p = pathlib.Path(__file__).absolute().parent


def get_path() -> str:
    return str(p)
