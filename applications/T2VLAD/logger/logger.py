import os
import logging
import logging.config
from pathlib import Path
from utils import read_json


def setup_logging(save_dir, log_config='logger/logger_config.json',
                  default_level=logging.INFO):
    """Setup logging configuration."""
    print(os.getcwd())
    log_config = Path(log_config)
    print(f"log config: {log_config} exists: {log_config.exists()}")
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir / handler['filename'])

        logging.config.dictConfig(config)
    else:
        print(f"Warning: logging configuration file is not found in {log_config}.")
        logging.basicConfig(level=default_level)
    return config["handlers"]["info_file_handler"]["filename"]
