import os
from utils.logger import configure_logger

logger = configure_logger(__name__)

def cleanup_output_folder(output_dir="output"):
    """
    Deletes all files in the specified output directory while keeping the directory itself intact.

    Args:
        output_dir (str): The path to the output directory.
    """
    if not os.path.exists(output_dir):
        logger.warning(f"The directory '{output_dir}' does not exist. Nothing to clean.")
        return

    try:
        files_removed = 0
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                files_removed += 1
                logger.info(f"Deleted file: {file_path}")

        if files_removed == 0:
            logger.info(f"No files found in '{output_dir}' to delete.")
        else:
            logger.info(f"Cleanup complete. {files_removed} file(s) deleted from '{output_dir}'.")
    except Exception as e:
        logger.error(f"An error occurred while cleaning up the output folder: {e}")
        raise

if __name__ == "__main__":
    cleanup_output_folder()
