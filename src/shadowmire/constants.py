import os
import re

USER_AGENT = "Shadowmire (https://github.com/taoky/shadowmire)"
LOCAL_DB_NAME = "local.db"
LOCAL_JSON_NAME = "local.json"
LOCAL_DB_SERIAL_NAME = "local.db.serial"

# Sentinel stored in LocalVersionKV.local.value when a project is present in the
# upstream project list but its project metadata endpoint returns not found.
PACKAGE_NOT_FOUND_SERIAL = -1

# Values reserved for LocalVersionKV.local.file_serial. A NULL file_serial is
# intentionally backwards compatible: it is interpreted as the row's metadata
# serial (value), so merely upgrading an existing database does not schedule IO.
PACKAGE_FILES_PENDING = -2
PACKAGE_FILES_METADATA_ONLY = -1

# Note that it's suggested to use only 3 workers for PyPI.
WORKERS = int(os.environ.get("SHADOWMIRE_WORKERS", "3"))
# Use threads to parallelize verification local IO.
IOWORKERS = int(os.environ.get("SHADOWMIRE_IOWORKERS", "2"))
# Avoid upstream issues causing too many packages to be removed from a plan.
MAX_DELETION = int(os.environ.get("SHADOWMIRE_MAX_DELETION", "50000"))
# Avoid permanently marking recently listed, temporarily unavailable packages.
IGNORE_THRESHOLD = int(os.environ.get("SHADOWMIRE_IGNORE_THRESHOLD", "10000"))

PRERELEASE_PATTERNS = (
    re.compile(r".+rc\d+$"),
    re.compile(r".+a(lpha)?\d+$"),
    re.compile(r".+b(eta)?\d+$"),
    re.compile(r".+dev\d+$"),
)
