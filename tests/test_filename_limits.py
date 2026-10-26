import errno
import json
from unittest.mock import Mock
from urllib.parse import quote

import pytest

from shadowmire.constants import PACKAGE_FILES_METADATA_ONLY, PACKAGE_NOT_FOUND_SERIAL
from shadowmire.database import LocalVersionKV
from shadowmire.filters import FileInclusionChecker, PackageInclusionChecker
from shadowmire.sync.base import SyncBase
from shadowmire.sync.plain_http import SyncPlainHTTP
from shadowmire.sync.pypi import SyncPyPI


def metadata(filename, project="demo", url_name=None):
    return {
        "info": {"name": project},
        "last_serial": 42,
        "releases": {
            "1.0": [
                {
                    "filename": filename,
                    "url": "https://files.example/packages/"
                    + quote(url_name or filename),
                    "digests": {"sha256": "abcd"},
                    "size": 5,
                }
            ]
        },
    }


def file_checker(exclude=()):
    return FileInclusionChecker((), exclude, False, False, None, 0)


@pytest.fixture(params=[SyncPyPI, SyncPlainHTTP])
def syncer(request, tmp_path):
    backend = object.__new__(request.param)
    db = LocalVersionKV(tmp_path / "local.db", tmp_path / "local.json")
    SyncBase.__init__(backend, tmp_path, db, sync_packages=True)
    backend.upstream = "https://mirror.example/"
    backend.session = Mock()
    backend.get_package_simple = Mock(return_value={"files": []})
    yield backend
    db.conn.close()


def provide_metadata(syncer, meta):
    if isinstance(syncer, SyncPyPI):
        syncer.get_package_metadata = Mock(return_value=meta)
    response = Mock(status_code=200)
    response.json.return_value = meta
    response.iter_content.return_value = [json.dumps(meta).encode()]
    syncer.session.get.return_value = response


@pytest.mark.parametrize("length", [240, 241, 246])
def test_filename_boundary(syncer, length):
    filename = "a" * (length - 4) + ".whl"
    provide_metadata(syncer, metadata(filename))
    syncer.get_package_simple.return_value = {
        "files": [{"filename": filename, "core-metadata": {"sha256": "abcd"}}]
    }

    serial = syncer.do_update("demo", file_checker(), True)

    if length == 240:
        assert serial == 42
        assert (syncer.packages_dir / filename).is_file()
        assert (syncer.packages_dir / (filename + ".metadata")).is_file()
        assert syncer.local_db.get("demo") == 42
    else:
        assert serial == PACKAGE_NOT_FOUND_SERIAL
        assert syncer.local_db.get("demo") == PACKAGE_NOT_FOUND_SERIAL
        assert syncer.local_db.dump_file_serials(skip_invalid=False) == {
            "demo": PACKAGE_FILES_METADATA_ONLY
        }
        assert not list(syncer.packages_dir.iterdir())
        assert not list(syncer.simple_dir.iterdir())
        assert not list(syncer.jsonmeta_dir.iterdir())
        syncer.get_package_simple.assert_not_called()
        # Plain HTTP fetches JSON into .new; neither backend downloads artifacts.
        assert syncer.session.get.call_count == int(isinstance(syncer, SyncPlainHTTP))


@pytest.mark.parametrize(
    "sync_files,included", [(True, True), (True, False), (False, True)]
)
def test_rejection_precedes_file_filters_and_download_policy(
    syncer, sync_files, included
):
    syncer.sync_packages = sync_files
    provide_metadata(syncer, metadata("a" * 241))
    assert syncer.do_update("demo", file_checker((".*",)), included) == -1
    assert syncer.local_db.get("demo") == -1


@pytest.mark.parametrize(
    "filename,url_name",
    [("é" * 121, None), ("ok.whl", "é" * 121), ("ok.whl", "a" * 241 + "/ok.whl")],
)
def test_encoded_filenames_and_decoded_url_components(syncer, filename, url_name):
    provide_metadata(syncer, metadata(filename, url_name=url_name))
    assert syncer.do_update("demo", file_checker(), True) == -1


@pytest.mark.parametrize("length", [241, 300])
def test_overlong_project_name_is_rejected_before_fetch(syncer, length):
    syncer.get_package_metadata = Mock(side_effect=AssertionError("unexpected fetch"))
    name = "a" * length
    assert syncer.do_update(name, file_checker(), True) == -1
    assert syncer.local_db.get(name) == -1
    syncer.get_package_metadata.assert_not_called()


def test_project_name_at_limit_is_accepted(syncer):
    name = "a" * 240
    provide_metadata(syncer, metadata("ok.whl", name))
    assert syncer.do_update(name, file_checker(), True) == 42
    assert (syncer.jsonmeta_dir / name).is_file()
    assert (syncer.simple_dir / name / "index.v1_json").is_file()


def test_rejection_removes_old_project_and_staging_files(syncer):
    project_dir = syncer.simple_dir / "demo"
    project_dir.mkdir()
    old_file = syncer.packages_dir / "old file.whl"
    old_file.write_bytes(b"old")
    old_metadata = old_file.with_name(old_file.name + ".metadata")
    old_metadata.write_bytes(b"metadata")
    # An old metadata-only index can also contain an unrepresentable path.
    (project_dir / "index.v1_json").write_text(
        json.dumps(
            {
                "files": [
                    {"url": "../../packages/old%20file.whl", "core-metadata": True},
                    {"url": "../../packages/" + "a" * 300, "core-metadata": True},
                ]
            }
        )
    )
    for suffix in ("", ".new", ".new.tmp", ".tmp"):
        (syncer.jsonmeta_dir / ("demo" + suffix)).write_text("{}")
    syncer.local_db.set_with_file_serial("demo", 1, 1)
    provide_metadata(syncer, metadata("a" * 241))

    assert syncer.do_update("demo", file_checker(), True) == -1

    assert not project_dir.exists()
    assert not old_file.exists()
    assert not old_metadata.exists()
    assert not list(syncer.jsonmeta_dir.iterdir())
    syncer.finalize(42)
    assert syncer.local_db.dump() == {}
    assert json.loads((syncer.basedir / "local.json").read_text()) == {}
    assert (
        json.loads((syncer.simple_dir / "index.v1_json").read_text())["projects"] == []
    )
    assert 'href="demo/"' not in (syncer.simple_dir / "index.v1_html").read_text()
    syncer.fetch_remote_versions = Mock(return_value=(43, {"demo": 43}))
    plan = syncer.determine_sync_plan(
        syncer.local_db.dump(skip_invalid=False), PackageInclusionChecker((), ())
    )
    assert plan.update == []
    assert syncer.do_sync_plan(plan, PackageInclusionChecker((), ()), file_checker())
    assert syncer.local_db.get("demo") == -1


def test_parallel_rejection_preserves_other_projects(syncer):
    bad = metadata("a" * 241, "bad")
    good = metadata("good.whl", "good")

    def get_meta(name):
        meta = bad if name == "bad" else good
        if isinstance(syncer, SyncPlainHTTP):
            (syncer.jsonmeta_dir / (name + ".new")).write_text(json.dumps(meta))
        return meta

    syncer.get_package_metadata = Mock(side_effect=get_meta)
    syncer.session.get.return_value = Mock(
        status_code=200, iter_content=Mock(return_value=[b"wheel"])
    )
    assert syncer.parallel_update(
        ["bad", "good"], PackageInclusionChecker((), ()), file_checker()
    )
    assert syncer.local_db.dump(skip_invalid=False) == {"bad": -1, "good": 42}
    assert syncer.local_db.dump_file_serials(skip_invalid=False) == {
        "bad": -1,
        "good": 42,
    }
    assert (syncer.packages_dir / "good.whl").read_bytes() == b"wheel"


@pytest.mark.parametrize("error", [errno.EACCES, errno.ENOSPC])
def test_other_filesystem_errors_propagate(syncer, monkeypatch, error):
    provide_metadata(syncer, metadata("a" * 241))
    monkeypatch.setattr(
        "shadowmire.sync.base.get_existing_hrefs",
        Mock(side_effect=OSError(error, "failure")),
    )
    with pytest.raises(OSError) as exc:
        syncer.do_update("demo", file_checker(), True)
    assert exc.value.errno == error
    assert syncer.local_db.get("demo") is None
