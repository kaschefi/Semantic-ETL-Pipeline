# src/utils/cache_cleaner.py
import os
import time
from src.utils.logger import logger


def clean_cache_directory(cache_dir: str, max_age_seconds: int = 3600) -> int:
    """
    Purges temporary extracted files from cache_dir older than max_age_seconds.
    Returns the count of deleted files.
    """
    if not os.path.exists(cache_dir):
        return 0

    now = time.time()
    deleted_count = 0

    try:
        for filename in os.listdir(cache_dir):
            if filename == ".gitkeep":
                continue

            file_path = os.path.join(cache_dir, filename)
            if os.path.isfile(file_path):
                file_age = now - os.path.getmtime(file_path)
                if file_age > max_age_seconds:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                    except Exception as err:
                        logger.warning(f"Failed to purge cached file {filename}: {err}")

        if deleted_count > 0:
            logger.info(f"Purged {deleted_count} stale cache files from {cache_dir}")

    except Exception as e:
        logger.error(f"Error during cache cleanup execution: {e}")

    return deleted_count
