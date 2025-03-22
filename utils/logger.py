import logging

def configure_logger(name=__name__, level=logging.INFO, log_format="%(asctime)s - %(levelname)s - %(message)s"):
    """
    Configures and returns a logger instance.

    Args:
        name (str): The name of the logger (usually the module name).
        level (int): The logging level (e.g., logging.INFO, logging.DEBUG).
        log_format (str): The format for log messages.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():  # Avoid adding multiple handlers
        handler = logging.StreamHandler()
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
