import logging
import socket
import xmlrpc.client
from http.client import HTTPConnection
from urllib.parse import urljoin

from .constants import USER_AGENT
from .errors import PackageNotFoundError
from .filesystem import normalize
from .http import create_requests_session

logger = logging.getLogger(__name__)


class CustomXMLRPCTransport(xmlrpc.client.Transport):
    """
    Set user-agent for xmlrpc.client
    """

    user_agent = USER_AGENT

    def make_connection(self, host: tuple[str, dict[str, str]] | str) -> HTTPConnection:
        conn = super().make_connection(host)
        if socket.getdefaulttimeout() is None:
            # By default conn.timeout is socket._GLOBAL_DEFAULT_TIMEOUT instead of None.
            # So here we check if default timeout is set, and if not, add a 2-min timeout
            conn.timeout = 120
        return conn


class PyPI:
    """
    Upstream which implements full PyPI APIs
    """

    host = "https://pypi.org"

    def __init__(self) -> None:
        self.xmlrpc_client = xmlrpc.client.ServerProxy(
            urljoin(self.host, "pypi"), transport=CustomXMLRPCTransport()
        )
        self.session = create_requests_session()

    def list_packages_with_serial(self, do_normalize: bool = True) -> dict[str, int]:
        logger.info(
            "Calling list_packages_with_serial() RPC, this requires some time..."
        )
        ret: dict[str, int] = self.xmlrpc_client.list_packages_with_serial()  # type: ignore
        if do_normalize:
            for key in list(ret.keys()):
                normalized_key = normalize(key)
                if normalized_key == key:
                    continue
                ret[normalized_key] = ret[key]
                del ret[key]
        return ret

    def changelog_last_serial(self) -> int:
        return self.xmlrpc_client.changelog_last_serial()  # type: ignore

    def get_package_metadata(self, package_name: str) -> dict:
        req = self.session.get(urljoin(self.host, f"pypi/{package_name}/json"))
        if req.status_code == 404:
            raise PackageNotFoundError
        return req.json()

    def get_package_simple(self, package_name: str) -> dict:
        # Based on PEP 691
        headers = {"Accept": "application/vnd.pypi.simple.v1+json"}
        req = self.session.get(
            urljoin(self.host, f"simple/{package_name}/"), headers=headers
        )
        # For incorrectly configured mirrors that do not return correct content-type
        # No need for dealing with application/vnd.pypi.simple.v1+html or text/html
        # Because most of them do not support PEP 658 so we don't need this
        if req.headers.get("Content-Type", "") != "application/vnd.pypi.simple.v1+json":
            raise PackageNotFoundError
        if req.status_code == 404:
            raise PackageNotFoundError
        return req.json()
