import json
import os
from pathlib import Path
import selectors
import signal
import sqlite3
import subprocess
import sys

import pytest
from click.testing import CliRunner

from shadowmire import cancellation
from shadowmire.cli import cli
from shadowmire.sync.base import Plan, SyncBase


@pytest.fixture(autouse=True)
def reset_cancellation():
    cancellation.reset()
    yield
    cancellation.reset()


@pytest.mark.parametrize(
    "iterator", [cancellation.cancellable_iter, cancellation.cancellable_completed]
)
def test_empty_iteration_observes_stop(iterator):
    cancellation.request_stop()
    with pytest.raises(cancellation.StopRequested):
        list(iterator([]))


def test_stop_during_last_removal_saves_state_without_finalizing(tmp_path, monkeypatch):
    from shadowmire.database import LocalVersionKV

    db = LocalVersionKV(tmp_path / "local.db", tmp_path / "local.json")
    db.set("old", 1)
    db.dump_json()
    db.conn.close()

    def remove(self, package_name, **kwargs):
        self.local_db.remove(package_name)
        cancellation.request_stop()

    monkeypatch.setattr(SyncBase, "do_remove", remove)
    monkeypatch.setattr(
        SyncBase, "determine_sync_plan", lambda *a, **kw: Plan(["old"], [], [], {}, 99)
    )
    monkeypatch.setattr("shadowmire.cli.get_syncer", lambda p, db, *a: SyncBase(p, db))
    result = CliRunner().invoke(cli, ["--repo", str(tmp_path), "sync"])
    assert result.exit_code == 1
    assert json.loads((tmp_path / "local.json").read_text()) == {}
    assert not (tmp_path / "local.db.serial").exists()
    assert not (tmp_path / "simple" / "index.v1_json").exists()


# Use real signals and real executors in a subprocess: os._exit must never run
# in the test runner. The worker remains blocked well beyond the exit deadline.
SCRIPT = """
import sys, time
from pathlib import Path
import shadowmire.cli as cli
from shadowmire.database import LocalVersionKV
from shadowmire.sync.base import SyncBase, Plan

repo = Path(sys.argv[1])
command = sys.argv[2]
mode = command
db = LocalVersionKV(repo / "local.db", repo / "local.json")
db.set("old", 1)
db.set("kept", 7)
db.dump_json()
db.conn.close()

class SlowSync(SyncBase):
    def determine_sync_plan(self, *a, **kw):
        return Plan(["old"], ["blocked"], [], {}, 99)

    def do_update(self, *a, **kw):
        assert kw.get("use_db", a[3] if len(a) > 3 else True) is False
        print("READY", flush=True)
        time.sleep(30)
        return 9

    def fetch_remote_versions(self):
        print("READY", flush=True)
        time.sleep(30)
        return 99, {}

if command == "planning":
    SlowSync.determine_sync_plan = SyncBase.determine_sync_plan
    command = "sync"
if command == "genlocal":
    (repo / "json").mkdir()
    (repo / "json" / "demo").write_text('{"last_serial": 9}')
    def slow_read(*args):
        print("READY", flush=True)
        time.sleep(30)
        return 9
    cli.get_local_serial = slow_read

cli.get_syncer = lambda p, db, sync_packages, *a: SlowSync(p, db, sync_packages)
cli.SyncPyPI = SlowSync
sys.argv = ["shadowmire", "--repo", str(repo), command]
if command == "do-update":
    sys.argv.append("blocked")
cli.main()
"""


@pytest.mark.parametrize(
    "command",
    ["sync", "planning", "genlocal", "do-update", "list-packages-with-serial"],
)
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_signal_exits_with_blocked_worker_and_saves_state(tmp_path, command, signum):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    proc = subprocess.Popen(
        [sys.executable, "-c", SCRIPT, str(tmp_path), command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=5), "worker did not start"
            assert proc.stdout.readline().strip() == "READY"
        proc.send_signal(signum)
        stdout, stderr = proc.communicate(timeout=5)
        assert proc.returncode == 1, (stdout, stderr)
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.communicate()

    expected = {"kept": 7} if command == "sync" else {"old": 1, "kept": 7}
    assert json.loads((tmp_path / "local.json").read_text()) == expected
    with sqlite3.connect(tmp_path / "local.db") as db:
        assert dict(db.execute("SELECT key, value FROM local")) == expected
    assert not (tmp_path / "local.db.serial").exists()


def test_manual_update_commits_worker_result_on_main_thread(tmp_path, monkeypatch):
    class SuccessfulSync(SyncBase):
        def do_update(self, name, checker, package_files_included, use_db=True):
            assert not use_db
            return 42

    monkeypatch.setattr(
        "shadowmire.cli.get_syncer", lambda p, db, *a: SuccessfulSync(p, db, True)
    )
    result = CliRunner().invoke(cli, ["--repo", str(tmp_path), "do-update", "demo"])
    assert result.exit_code == 0, result.output
    with sqlite3.connect(tmp_path / "local.db") as db:
        assert db.execute(
            "SELECT value, file_serial FROM local WHERE key='demo'"
        ).fetchone() == (42, 42)


def test_completed_iterator_preserves_errors_and_deduplicates():
    from concurrent.futures import Future

    ok = Future()
    ok.set_result(42)
    failed = Future()
    failed.set_exception(ValueError("worker failed"))
    completed = list(cancellation.cancellable_completed([ok, failed, ok]))
    assert set(completed) == {ok, failed}
    assert len(completed) == 2
    assert ok.result() == 42
    with pytest.raises(ValueError, match="worker failed"):
        failed.result()
