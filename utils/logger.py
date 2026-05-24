import logging
import os

def get_logger(name, log_file):

    os.makedirs(
        os.path.dirname(log_file),
        exist_ok=True
    )

    logger = logging.getLogger(name)

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s - %(message)s'
    )

    fh = logging.FileHandler(log_file)

    fh.setLevel(logging.DEBUG)

    fh.setFormatter(formatter)

    sh = logging.StreamHandler()

    sh.setLevel(logging.INFO)

    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger