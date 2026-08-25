"""Utilidades compartilhadas entre estágios."""
import os
import yaml
from pathlib import Path
from sqlalchemy import create_engine

DATA = Path("/data")
RAW, DERIVED, OUTPUTS = DATA / "raw", DATA / "derived", DATA / "outputs"


def engine():
    return create_engine(os.environ["DATABASE_URL"], pool_pre_ping=True)


def config() -> dict:
    with open("config/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
