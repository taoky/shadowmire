import functools
import logging
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter, Retry

from .constants import USER_AGENT
from .filesystem import overwrite

logger = logging.getLogger("shadowmire")


def create_requests_session() -> requests.Session:
    s = requests.Session()
    # hardcode 1min timeout for connect & read for now
    # https://requests.readthedocs.io/en/latest/user/advanced/#timeouts
    # A hack to overwrite get() method
    s.get_orig, s.get = s.get, functools.partial(s.get, timeout=(60, 60))  # type: ignore
    retries = Retry(total=3, backoff_factor=0.1)
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def download(
    session: requests.Session, url: str, dest: Path
) -> tuple[bool, requests.Response | None]:
    try:
        resp = session.get(url, allow_redirects=True)
    except requests.RequestException:
        logger.warning("download %s failed with exception", exc_info=True)
        return False, None
    if resp.status_code >= 400:
        logger.warning(
            "download %s failed with status %s, skipping this package",
            url,
            resp.status_code,
        )
        return False, resp
    with overwrite(dest, "wb") as f:
        f.write(resp.content)
    return True, resp
