import logging
import sys

def get_basic_logger():
    if "logger" in logging.Logger.manager.loggerDict.keys():
        return logging.getLogger("basic")
    else:
        logger = logging.getLogger("basic")
        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
