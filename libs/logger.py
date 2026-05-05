import logging

def setup_logging(out_log_file=None):
    logging.root.setLevel(level=logging.INFO)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level=logging.INFO)

    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    stream_handler = logging.StreamHandler()
    logging.root.addHandler(stream_handler)

    if out_log_file is not None:
        file_handler = logging.FileHandler(out_log_file)
        logging.root.addHandler(file_handler)

    for handler in logging.root.handlers:
        handler.setFormatter(formatter)