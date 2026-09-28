
import logging
import os
import os.path as osp

import torch.distributed as dist
from termcolor import colored

logger_name = None


def get_root_logger(name=None, log_file=None, log_level=logging.INFO, file_mode='w'):
    """Get root logger with file and stream handlers.

    Args:
        name (str): Logger name.
        log_file (str | None): Log filename. If specified, a FileHandler will be added.
        log_level (int): Log level. Default: logging.INFO.
        file_mode (str): Mode to open log file. Default: 'w'.

    Returns:
        logging.Logger: The root logger.
    """
    logger = logging.getLogger(name)
    # if the logger has been initialized, just return it
    if logger.hasHandlers():
        return logger

    logger.setLevel(log_level)

    # create stream handler (console output)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    logger.addHandler(stream_handler)

    # create file handler if log_file is specified
    if log_file is not None:
        os.makedirs(osp.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, file_mode)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


def get_logger(cfg=None, log_level=logging.INFO):
    global logger_name
    if cfg is None:
        return get_root_logger(logger_name)

    # creating logger
    name = cfg.model.name
    output = cfg.output
    logger_name = name

    # Only rank 0 writes to file to avoid IO contention
    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()

    # Only create file handler on rank 0
    log_file = osp.join(output, 'run.log') if rank == 0 else None
    logger = get_root_logger(name, log_file, log_level=log_level, file_mode='a')

    ### stop duplicate logger, see https://zhuanlan.zhihu.com/p/487524917 ###
    logger.propagate = False


    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') \
        + colored('(%(filename)s %(lineno)d)', 'yellow') + ': %(levelname)s %(message)s'

    for handler in logger.handlers:

        if isinstance(handler, logging.FileHandler):
            handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))

        elif isinstance(handler, logging.StreamHandler):
            handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))

    return logger
