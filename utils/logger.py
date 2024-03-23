import logging
import os
import sys
from utils.args import args


def get_logger(name, path, mode):
    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s %(asctime)s: %(message)s")

    fc = logging.FileHandler(path, mode=mode, encoding="utf-8")
    fc.setFormatter(formatter)
    logger.addHandler(fc)

    fc = logging.StreamHandler(sys.stdout)
    fc.setFormatter(formatter)
    logger.addHandler(fc)
    return logger


if not os.path.exists("logging"):
    os.makedirs("logging")
if not os.path.exists("logging/result"):
    os.makedirs("logging/result")

result_logger = get_logger("result", args.result_log, "a+")
run_logger = get_logger("run", args.run_log, "a+")
