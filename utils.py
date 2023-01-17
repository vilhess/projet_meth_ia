import argparse
import subprocess
import sys


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# hack to install packages during execution
# if they are not already installed
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
