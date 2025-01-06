from pathlib import Path
import logging

def ensure_dir(directory: Path) -> None:
    """Ensure directory exists, create if not exists"""
    try:
        if not directory.exists():
            directory.mkdir(parents=True)
            logging.info(f"Created directory: {directory}")
    except Exception as e:
        logging.error(f"Error creating directory {directory}: {str(e)}")
        raise 