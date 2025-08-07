import os
import sys
import time
import os.path as osp
from datetime import datetime

from .tools import mkdir_if_missing

__all__ = ["Logger", "setup_logger"]


class Logger:
    """
    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py>`

    Write console output to external text file with timestamp and filename.

    Args:
        fpath (str): directory to save logging file.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        self.last_filename = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, "w", encoding='utf-8')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def _get_caller_info(self):
        """Get the calling file information"""
        import inspect
        frame = inspect.currentframe()
        try:
            # Go up the call stack to find the calling file
            while frame:
                frame = frame.f_back
                if frame and frame.f_code.co_filename != __file__:
                    filename = os.path.basename(frame.f_code.co_filename).replace('.py', '')
                    return filename
        finally:
            del frame
        return "unknown"

    def _format_message(self, msg):
        """Format message with timestamp and filename"""
        if not msg.strip():  # Skip empty messages
            return msg

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = self._get_caller_info()

        # Only add timestamp if it's not already there
        if not msg.startswith(f"[{timestamp[:10]}"):  # Check if timestamp already exists
            return f"[{timestamp}] [{filename}] {msg}"
        return msg

    def write(self, msg):
        formatted_msg = self._format_message(msg)
        self.console.write(formatted_msg)
        if self.file is not None:
            self.file.write(formatted_msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def setup_logger(output=None):
    if output is None:
        return

    if output.endswith(".txt") or output.endswith(".log"):
        fpath = output
    else:
        fpath = osp.join(output, "log.txt")

    if osp.exists(fpath):
        # make sure the existing log file is not over-written
        fpath += time.strftime("-%Y-%m-%d-%H-%M-%S")

    sys.stdout = Logger(fpath)
