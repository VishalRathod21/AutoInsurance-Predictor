import logging
import os
import functools
from logging.handlers import RotatingFileHandler
from from_root import from_root
from datetime import datetime
from typing import Callable, Any

# Constants for log configuration
LOG_DIR = 'logs'
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3  # Number of backup log files to keep

# Construct log file path
log_dir_path = os.path.join(from_root(), LOG_DIR)
os.makedirs(log_dir_path, exist_ok=True)
log_file_path = os.path.join(log_dir_path, LOG_FILE)

def configure_logger():
    """
    Configures logging with a rotating file handler and a console handler.
    """
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Define formatter
    formatter = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")

    # File handler with rotation
    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def log_function_call(logger: logging.Logger = None):
    """
    A decorator that logs the function call details and execution time.
    
    Args:
        logger: Logger instance to use. If None, creates a new logger.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get logger if not provided
            func_logger = logger or logging.getLogger(func.__module__)
            
            # Log function call
            func_logger.debug(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
            
            try:
                # Execute the function
                result = func(*args, **kwargs)
                func_logger.debug(f"Function {func.__name__} executed successfully")
                return result
            except Exception as e:
                func_logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                raise
                
        return wrapper
    return decorator

# Configure the logger
configure_logger()