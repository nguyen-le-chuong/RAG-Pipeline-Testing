import logging

def setup_logging(log_level):
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO), format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)