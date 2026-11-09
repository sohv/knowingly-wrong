# console plus output_dir/run.log logging for experiment scripts.

import logging
from pathlib import Path

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(output_dir: str | Path, level: int = logging.INFO) -> Path:
    log_path = Path(output_dir) / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(), logging.FileHandler(log_path)]
    logging.basicConfig(level=level, format=FORMAT, handlers=handlers, force=True)
    # keep http client chatter out of the run log
    for name in ("httpx", "httpx2", "openai", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return log_path
