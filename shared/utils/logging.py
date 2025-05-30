import logging
import sys
from typing import Optional
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)

        if hasattr(record, 'extra'):
            log_obj.update(record.extra)

        return json.dumps(log_obj)


def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    from .config import Config
    config = Config()

    level = level or config.get('logging.level', 'INFO')
    logger.setLevel(getattr(logging, level))

    handler = logging.StreamHandler(sys.stdout)

    if config.get('logging.json', False):
        formatter = JSONFormatter()
    else:
        format_string = config.get('logging.format',
                                   '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter(format_string)

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger