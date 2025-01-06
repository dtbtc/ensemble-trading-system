import logging
from pathlib import Path
import sys
from datetime import datetime

def setup_logger(log_dir: Path = None):
    """Setup logger"""
    if log_dir is None:
        log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'run_{timestamp}.log'
    
    # Configure log format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger 