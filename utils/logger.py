import os
import sys
from loguru import logger

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Remove default Loguru handler
logger.remove()

# Add sink for terminal (stdout)
logger.add(sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# Add sink for log file
logger.add("logs/output.log",
           rotation="10 MB",
           retention="7 days",
           compression="zip",
           level="DEBUG",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

# Optional: Redirect stdout and stderr to Loguru
# Uncomment if you want to capture all print/errors via logger
# sys.stdout = StreamToLoguru("DEBUG")
# sys.stderr = StreamToLoguru("ERROR")
print(f"Terminal output saved to logs/output.log")

logger.info("")
logger.info("")
logger.info("------------------------------------------------------------------\n"
"                             ---      Logger initialized and stderr are now redirected      ---\n"
"                             ------------------------------------------------------------------")
