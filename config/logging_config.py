"""
FasalSetu — Logging Configuration
Sets up rotating file logs and a separate compliance audit log.

Usage (in main.py or any entry point):
    from config.logging_config import setup_logging
    setup_logging()
"""

import logging
import logging.config
import os
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "compliance": {
            "format": "%(asctime)s [COMPLIANCE] %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S",
        },
        "json": {
            "()": "config.logging_config.JsonFormatter",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "app_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": str(LOG_DIR / "fasalsetu.log"),
            "maxBytes": 10 * 1024 * 1024,  # 10 MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
        "compliance_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "compliance",
            "filename": str(LOG_DIR / "compliance.log"),
            "maxBytes": 50 * 1024 * 1024,  # 50 MB — compliance logs kept longer
            "backupCount": 20,
            "encoding": "utf-8",
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "standard",
            "filename": str(LOG_DIR / "errors.log"),
            "maxBytes": 5 * 1024 * 1024,
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        # Application logger
        "fasalsetu": {
            "handlers": ["console", "app_file"],
            "level": "DEBUG",
            "propagate": False,
        },
        # Compliance — goes to its own file AND console
        "guardrail": {
            "handlers": ["console", "compliance_file"],
            "level": "INFO",
            "propagate": False,
        },
        # Agent-specific loggers
        "agents.soil": {
            "handlers": ["app_file"],
            "level": "DEBUG",
            "propagate": True,
        },
        "agents.disease": {
            "handlers": ["app_file"],
            "level": "DEBUG",
            "propagate": True,
        },
        "agents.market": {
            "handlers": ["app_file"],
            "level": "DEBUG",
            "propagate": True,
        },
        "agents.scheme": {
            "handlers": ["app_file"],
            "level": "DEBUG",
            "propagate": True,
        },
        "agents.weather": {
            "handlers": ["app_file"],
            "level": "DEBUG",
            "propagate": True,
        },
        "agents.voice": {
            "handlers": ["app_file"],
            "level": "DEBUG",
            "propagate": True,
        },
        "agents.offline": {
            "handlers": ["app_file"],
            "level": "DEBUG",
            "propagate": True,
        },
        # Silence noisy third-party loggers
        "httpx": {"level": "WARNING"},
        "httpcore": {"level": "WARNING"},
        "chromadb": {"level": "WARNING"},
        "langchain": {"level": "WARNING"},
        "urllib3": {"level": "WARNING"},
        "uvicorn.access": {
            "handlers": ["app_file"],
            "level": "INFO",
            "propagate": False,
        },
    },
    "root": {
        "handlers": ["console", "app_file", "error_file"],
        "level": "INFO",
    },
}


class JsonFormatter(logging.Formatter):
    """Optional JSON formatter for structured log ingestion (e.g. ELK, CloudWatch)."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        from datetime import datetime, timezone

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """
    Call once at application startup.

    Args:
        level: Root log level override ("DEBUG", "INFO", "WARNING", "ERROR")
        json_logs: If True, switches console handler to JSON format (useful in prod)
    """
    config = dict(LOGGING_CONFIG)

    # Override root level
    config["root"]["level"] = level.upper()

    # Switch to JSON on console if requested
    if json_logs:
        config["handlers"]["console"]["formatter"] = "json"

    logging.config.dictConfig(config)
    logger = logging.getLogger("fasalsetu")
    logger.info(
        "Logging initialised | level=%s | log_dir=%s | json=%s",
        level, LOG_DIR, json_logs,
    )


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper. Use in any module: logger = get_logger(__name__)"""
    return logging.getLogger(name)
