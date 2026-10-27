import json

import pytest

from shadowmire.database import LocalVersionKV
from shadowmire.filters import FileInclusionChecker, PackageInclusionChecker
from shadowmire.sync.base import SyncBase


@pytest.fixture
def syncer(tmp_path, monkeypatch):
    db = LocalVersionKV(tmp_path / "local.db", tmp_path / "local.json")
    syncer = SyncBase(tmp_path, db)
    serials = {"alpha": 1, "beta": 2, "gamma": 3}
    monkeypatch.setattr(
        syncer, "do_update", lambda name, *args, **kwargs: serials[name]
    )
    yield syncer
    db.conn.close()


def update_all(syncer):
    return syncer.parallel_update(
        ["alpha", "beta", "gamma"],
        PackageInclusionChecker((), ()),
        FileInclusionChecker((), (), False, False, None, 0),
    )


def test_no_dump_before_interval(syncer, monkeypatch):
    monkeypatch.setattr("shadowmire.sync.base.LOCAL_JSON_DUMP_INTERVAL", 3600)
    assert update_all(syncer)
    assert not syncer.local_db.jsonpath.exists()
    assert syncer.local_db.dump() == {"alpha": 1, "beta": 2, "gamma": 3}


def test_dump_after_interval(syncer, monkeypatch):
    monkeypatch.setattr("shadowmire.sync.base.LOCAL_JSON_DUMP_INTERVAL", 0)
    assert update_all(syncer)
    text = syncer.local_db.jsonpath.read_text()
    assert json.loads(text) == {"alpha": 1, "beta": 2, "gamma": 3}
    assert "\n" not in text
