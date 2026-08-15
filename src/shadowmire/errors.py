import logging
import sys
from concurrent.futures import Future
from typing import Any, NoReturn

logger = logging.getLogger("shadowmire")


class PackageNotFoundError(Exception):
    pass


class ExitProgramException(Exception):
    pass


def exit_with_futures(futures: dict[Future[Any], Any]) -> NoReturn:
    logger.info("Exiting...")
    for future in futures:
        future.cancel()
    sys.exit(1)
