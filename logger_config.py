import logging
import sys

def setup_logger():
    """Sets up the logging configuration for the application."""
    # Create a logger
    logger = logging.getLogger(__name__)  # Use __name__ to get the logger for the current module
    logger.setLevel(logging.DEBUG)  # Set the logging level

    # Create a handler that writes log messages to the console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)  # Set the handler's level

    # Create a formatter that defines the format of log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    return logger

# Call this function to set up logging.  You'll call it in main.py
logger = setup_logger()
if __name__ == "__main__":
    #This is for testing the logger
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")