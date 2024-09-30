import logging

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """Configura o logger."""
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# Exemplo de uso
api_logger = setup_logger("api_logger", "../../logs/api/api.log")
