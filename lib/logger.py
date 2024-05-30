import os
import logging
from datetime import datetime
import sys


def get_logger(log_dir, name, log_filename=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if log_filename is None:
        log_filename = datetime.now().strftime('%m-%d-%H-%M') + ".txt"
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)

    # Add console handler
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info('Log directory: %s', log_dir)

    return logger


if __name__ == '__main__':
    time = datetime.now().strftime('%m-%d-%H-%M') + ".txt"
    print(time)
