import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from shadowmire.http import DOWNLOAD_CHUNK_SIZE, create_requests_session, download

BODY = bytes(range(256)) * (DOWNLOAD_CHUNK_SIZE // 256 * 3 + 7)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/missing":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        if self.path == "/truncated":
            self.wfile.write(BODY[: len(BODY) // 2])
            self.close_connection = True
            return
        self.wfile.write(BODY)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def base_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def test_download_writes_body(tmp_path, base_url):
    dest = tmp_path / "file.whl"
    success, resp = download(create_requests_session(), f"{base_url}/ok", dest)
    assert success
    assert resp is not None and resp.status_code == 200
    assert dest.read_bytes() == BODY
    assert not (tmp_path / "file.whl.tmp").exists()


def test_download_http_error_returns_response(tmp_path, base_url):
    dest = tmp_path / "file.whl"
    success, resp = download(create_requests_session(), f"{base_url}/missing", dest)
    assert not success
    assert resp is not None and resp.status_code == 404
    assert not dest.exists()


def test_download_truncated_body_leaves_no_file(tmp_path, base_url):
    dest = tmp_path / "file.whl"
    success, resp = download(create_requests_session(), f"{base_url}/truncated", dest)
    assert not success
    assert resp is None
    assert not dest.exists()
    assert not (tmp_path / "file.whl.tmp").exists()
