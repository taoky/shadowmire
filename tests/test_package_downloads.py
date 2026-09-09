"""Distribution and PEP 658 sidecar downloads must be independently resumable."""

import json
from unittest.mock import Mock

import pytest
import requests

from shadowmire.constants import PACKAGE_FILES_PENDING, PACKAGE_NOT_FOUND_SERIAL
from shadowmire.database import LocalVersionKV
from shadowmire.sync import plain_http, pypi

WHEEL = b"existing wheel contents"
METADATA = b"Metadata-Version: 2.1\nName: demo\nVersion: 1.0\n"
FILENAME = "demo-1.0-py3-none-any.whl"
PATH = f"packages/ab/cd/{FILENAME}"
PYPI_URL = f"https://files.pythonhosted.org/{PATH}"
MIRROR_URL = f"https://mirror.example/{PATH}"


@pytest.fixture(params=[pypi, plain_http], ids=["pypi", "plain-http"])
def sync_case(request, tmp_path, monkeypatch):
    module = request.param
    db = LocalVersionKV(tmp_path / "local.db", tmp_path / "local.json")
    if module is pypi:
        syncer = module.SyncPyPI(tmp_path, db, sync_packages=True)
        url = PYPI_URL
    else:
        syncer = module.SyncPlainHTTP(
            "https://mirror.example/", tmp_path, db, sync_packages=True
        )
        url = MIRROR_URL
    meta = {
        "info": {"name": "demo"},
        "last_serial": 42,
        "releases": {
            "1.0": [
                {
                    "filename": FILENAME,
                    "url": PYPI_URL,
                    "size": len(WHEEL),
                    "digests": {"sha256": "abc"},
                }
            ]
        },
    }
    simple = {"files": [{"filename": FILENAME, "core-metadata": {"sha256": "def"}}]}

    def get_meta(name):
        # Plain HTTP stages the JSON response before publishing it.
        (syncer.jsonmeta_dir / f"{name}.new").write_text(json.dumps(meta))
        return meta

    monkeypatch.setattr(syncer, "get_package_metadata", get_meta)
    monkeypatch.setattr(syncer, "get_package_simple", lambda name: simple)
    checker = Mock()
    checker.has_rules.return_value = False
    checker.includes_package_files.return_value = True
    checker.get_filtered_meta.side_effect = lambda name, meta: meta
    dest = tmp_path / PATH
    dest.parent.mkdir(parents=True)
    responses = {}
    calls = []

    def download(session, requested_url, target):
        calls.append(requested_url)
        status = responses.get(requested_url, 200)
        if isinstance(status, BaseException):
            raise status
        if status is None:
            return False, None
        response = requests.Response()
        response.status_code = status
        if status >= 400:
            return False, response
        target.write_bytes(METADATA if requested_url.endswith(".metadata") else WHEEL)
        return True, response

    monkeypatch.setattr(module, "download", download)
    return {
        "run": lambda: syncer.do_update("demo", checker, package_files_included=True),
        "dest": dest,
        "sidecar": dest.with_name(dest.name + ".metadata"),
        "url": url,
        "calls": calls,
        "responses": responses,
        "db": db,
        "simple": simple,
        "meta": meta,
        "syncer": syncer,
        "checker": checker,
    }


def test_missing_sidecar_preserves_existing_wheel(sync_case):
    case = sync_case
    case["dest"].write_bytes(WHEEL)
    before = case["dest"].stat()

    assert case["run"]() == 42
    assert case["calls"] == [case["url"] + ".metadata"]
    assert case["dest"].read_bytes() == WHEEL
    assert case["dest"].stat().st_mtime_ns == before.st_mtime_ns
    assert case["sidecar"].read_bytes() == METADATA
    assert case["db"].get("demo") == 42

    # A subsequent update must not fetch either file again.
    case["calls"].clear()
    assert case["run"]() == 42
    assert case["calls"] == []


def test_interrupted_sidecar_download_resumes_without_rewriting_wheel(sync_case):
    case = sync_case
    case["responses"][case["url"] + ".metadata"] = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        case["run"]()
    assert case["calls"] == [case["url"], case["url"] + ".metadata"]
    assert case["dest"].read_bytes() == WHEEL
    assert not case["sidecar"].exists()
    assert case["db"].get("demo") is None
    before = case["dest"].stat().st_mtime_ns

    case["responses"].clear()
    case["calls"].clear()
    assert case["run"]() == 42
    assert case["calls"] == [case["url"] + ".metadata"]
    assert case["dest"].stat().st_mtime_ns == before
    assert case["db"].get("demo") == 42


@pytest.mark.parametrize("artifact", [None, b"truncated"])
@pytest.mark.parametrize("sidecar_exists", [False, True])
def test_missing_or_wrong_size_wheel_is_repaired(sync_case, artifact, sidecar_exists):
    case = sync_case
    if artifact is not None:
        case["dest"].write_bytes(artifact)
    if sidecar_exists:
        case["sidecar"].write_bytes(METADATA)

    assert case["run"]() == 42
    expected = [case["url"]]
    if not sidecar_exists:
        expected.append(case["url"] + ".metadata")
    assert case["calls"] == expected
    assert case["dest"].read_bytes() == WHEEL
    assert case["sidecar"].read_bytes() == METADATA


def test_no_sidecar_advertised_does_not_request_one(sync_case):
    case = sync_case
    case["simple"]["files"][0]["core-metadata"] = False
    assert case["run"]() == 42
    assert case["calls"] == [case["url"]]
    assert not case["sidecar"].exists()


def test_unknown_size_preserves_existing_wheel(sync_case):
    case = sync_case
    case["meta"]["releases"]["1.0"][0]["size"] = -1
    case["dest"].write_bytes(WHEEL)
    assert case["run"]() == 42
    assert case["calls"] == [case["url"] + ".metadata"]


@pytest.mark.parametrize("status", [500, None])
def test_failed_wheel_does_not_publish_serial_or_fetch_sidecar(sync_case, status):
    case = sync_case
    case["responses"][case["url"]] = status
    assert case["run"]() is None
    assert case["calls"] == [case["url"]]
    assert case["db"].get("demo") is None
    assert not case["sidecar"].exists()


def test_failed_sidecar_keeps_wheel_and_can_be_retried(sync_case):
    case = sync_case
    case["dest"].write_bytes(WHEEL)
    case["responses"][case["url"] + ".metadata"] = 500
    assert case["run"]() is None
    assert case["db"].get("demo") is None
    assert case["calls"] == [case["url"] + ".metadata"]
    assert case["dest"].read_bytes() == WHEEL
    assert not case["sidecar"].exists()
    case["responses"].clear()
    case["calls"].clear()
    assert case["run"]() == 42
    assert case["calls"] == [case["url"] + ".metadata"]
    assert case["sidecar"].read_bytes() == METADATA


@pytest.mark.parametrize("status", [500, 404, None])
@pytest.mark.parametrize("previous", [None, (41, 41), (42, PACKAGE_FILES_PENDING)])
def test_failed_sidecar_preserves_publication_and_retries_same_upstream_serial(
    sync_case, monkeypatch, status, previous
):
    case = sync_case
    syncer, db, checker = case["syncer"], case["db"], case["checker"]
    published = [syncer.jsonmeta_dir / "demo"] + [
        syncer.simple_dir / "demo" / name
        for name in ("index.html", "index.v1_html", "index.v1_json")
    ]
    if previous is not None:
        db.set_with_file_serial("demo", *previous)
        old_meta = dict(case["meta"], last_serial=previous[0])
        published[0].write_text(json.dumps(old_meta))
        published[1].parent.mkdir(parents=True)
        syncer.write_meta_to_simple(published[1].parent, old_meta, {})
        case["dest"].write_bytes(WHEEL)
    before = {path: path.read_bytes() if path.exists() else None for path in published}
    old_serials = db.dump(skip_invalid=False)
    old_file_serials = db.dump_file_serials()
    monkeypatch.setattr(syncer, "fetch_remote_versions", lambda: (42, {"demo": 42}))

    def plan():
        return syncer.determine_sync_plan(
            db.dump(skip_invalid=False),
            checker,
            local_file_serials=db.dump_file_serials(),
        )

    case["responses"][case["url"] + ".metadata"] = status
    # If the mirror returns 404, its PyPI fallback must also fail for this case.
    case["responses"][PYPI_URL + ".metadata"] = status
    assert plan().update == ["demo"]
    assert syncer.do_sync_plan(plan(), checker, checker) is False
    assert db.dump(skip_invalid=False) == old_serials
    assert db.dump_file_serials() == old_file_serials
    assert {
        path: path.read_bytes() if path.exists() else None for path in published
    } == before
    assert case["dest"].read_bytes() == WHEEL
    assert not case["sidecar"].exists()
    if case["url"] == MIRROR_URL and status == 404:
        assert case["calls"][-2:] == [MIRROR_URL + ".metadata", PYPI_URL + ".metadata"]
    wheel_mtime = case["dest"].stat().st_mtime_ns

    case["responses"].clear()
    case["calls"].clear()
    assert plan().update == ["demo"]
    assert syncer.do_sync_plan(plan(), checker, checker) is True
    assert case["calls"] == [case["url"] + ".metadata"]
    assert case["dest"].stat().st_mtime_ns == wheel_mtime
    assert db.get("demo") == 42
    assert db.dump_file_serials()["demo"] == 42
    assert case["sidecar"].read_bytes() == METADATA
    assert b"data-core-metadata" in published[1].read_bytes()
    assert plan().update == []


@pytest.mark.parametrize("failure", [None, RuntimeError("download failed"), "none"])
def test_parallel_update_reports_failures_and_preserves_successful_results(
    sync_case, monkeypatch, failure
):
    case = sync_case
    results = {"good": 42, "removed": PACKAGE_NOT_FOUND_SERIAL}
    if failure != "none":
        results["failed"] = failure

    def update(name, *args):
        result = results[name]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(case["syncer"], "do_update", update)
    assert case["syncer"].parallel_update(
        list(results), case["checker"], case["checker"]
    ) is (failure == "none")
    assert case["db"].get("good") == 42
    assert case["db"].get("removed") == PACKAGE_NOT_FOUND_SERIAL
    assert case["db"].get("failed") is None


def test_plain_http_missing_sidecar_falls_back_without_fetching_wheel(sync_case):
    case = sync_case
    if case["url"] != MIRROR_URL:
        pytest.skip("plain HTTP fallback")
    case["dest"].write_bytes(WHEEL)
    case["responses"][MIRROR_URL + ".metadata"] = 404
    assert case["run"]() == 42
    assert case["calls"] == [MIRROR_URL + ".metadata", PYPI_URL + ".metadata"]
    assert case["dest"].read_bytes() == WHEEL
    assert case["sidecar"].read_bytes() == METADATA


def test_plain_http_missing_wheel_falls_back_to_pypi(sync_case):
    case = sync_case
    if case["url"] != MIRROR_URL:
        pytest.skip("plain HTTP fallback")
    case["responses"][MIRROR_URL] = 404
    assert case["run"]() == 42
    assert case["calls"] == [MIRROR_URL, PYPI_URL, PYPI_URL + ".metadata"]
    assert case["dest"].read_bytes() == WHEEL
    assert case["sidecar"].read_bytes() == METADATA
