"""
Task 3 - Demonstrate the five log levels.

Run twice and compare (screenshot both for the report):
  python log_levels_demo.py               -> DEBUG line is hidden (level=INFO)
  python log_levels_demo.py --debug       -> all five lines appear
"""
import logging
import sys

from logger_config import logger

if "--debug" in sys.argv:
    logger.setLevel(logging.DEBUG)
    print(">>> logger level set to DEBUG\n")
else:
    print(">>> logger level is INFO (DEBUG messages are filtered out)\n")

logger.debug("detailed diagnostics for developers")
logger.info("normal event - service handled a request")
logger.warning("unexpected input but still handled")
logger.error("a request failed")
logger.critical("service cannot continue")
