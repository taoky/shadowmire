import json
import sqlite3
from unittest.mock import Mock

from shadowmire.constants import (
    PACKAGE_FILES_METADATA_ONLY,
    PACKAGE_FILES_PENDING,
)
from shadowmire.database import LocalVersionKV
from shadowmire.filters import (
    PACKAGE_FILTER,
    PackageInclusionChecker,
)
from shadowmire.sync.base import SyncBase


class StateSync(SyncBase):
    def __init__(self, basedir, local_db, remote, sync_packages):
        self.remote = remote
        self.updates = []
        super().__init__(basedir, local_db, sync_packages)

    def fetch_remote_versions(self):
        return 10, self.remote

    def do_update(
        self,
        package_name,
        file_inclusion_checker,
        package_files_included,
        use_db=True,
    ):
        self.updates.append((package_name, package_files_included))
        return self.remote[package_name]


def checker():
    return PackageInclusionChecker(
        include=(),
        exclude=(),
        package_filters=(
            PACKAGE_FILTER.convert("include:^popular$", None, None),
            PACKAGE_FILTER.convert("metadata-only:.*", None, None),
        ),
    )


def unfiltered_checker():
    return PackageInclusionChecker(include=(), exclude=())


def write_simple_project(basedir, package_name, filename):
    package_simple_dir = basedir / "simple" / package_name
    package_simple_dir.mkdir(parents=True)
    (package_simple_dir / "index.v1_json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "url": f"../../packages/{filename}",
                        "core-metadata": False,
                    }
                ]
            }
        )
    )
    return basedir / "packages" / filename


def test_old_database_migrates_without_changing_json_contract(tmp_path):
    db_path = tmp_path / "local.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE local(key TEXT PRIMARY KEY, value INT NOT NULL)")
    conn.execute("INSERT INTO local VALUES ('demo', 42)")
    conn.commit()
    conn.close()

    local_db = LocalVersionKV(db_path, tmp_path / "local.json")

    assert local_db.dump() == {"demo": 42}
    assert local_db.dump_file_serials() == {"demo": None}
    local_db.dump_json()
    assert json.loads((tmp_path / "local.json").read_text()) == {"demo": 42}


def test_null_file_serial_is_effectively_the_metadata_serial(tmp_path, monkeypatch):
    local_db = LocalVersionKV(tmp_path / "local.db", tmp_path / "local.json")
    local_db.set("popular", 1)
    syncer = StateSync(tmp_path, local_db, {"popular": 1}, sync_packages=True)
    get_existing_hrefs = Mock(side_effect=AssertionError("unexpected simple IO"))
    monkeypatch.setattr("shadowmire.sync.base.get_existing_hrefs", get_existing_hrefs)

    plan = syncer.determine_sync_plan(
        local_db.dump(skip_invalid=False),
        checker(),
        local_file_serials=local_db.dump_file_serials(skip_invalid=False),
    )

    assert plan.update == []
    assert plan.package_remove == []
    get_existing_hrefs.assert_not_called()


def test_null_metadata_only_state_does_not_trigger_upgrade_io(tmp_path, monkeypatch):
    local_db = LocalVersionKV(tmp_path / "local.db", tmp_path / "local.json")
    local_db.set("other", 1)
    syncer = StateSync(tmp_path, local_db, {"other": 1}, sync_packages=True)
    get_existing_hrefs = Mock(side_effect=AssertionError("unexpected simple IO"))
    monkeypatch.setattr("shadowmire.sync.base.get_existing_hrefs", get_existing_hrefs)

    plan = syncer.determine_sync_plan(
        local_db.dump(skip_invalid=False),
        checker(),
        local_file_serials=local_db.dump_file_serials(skip_invalid=False),
    )

    assert plan.update == []
    assert plan.package_remove == []
    get_existing_hrefs.assert_not_called()


def test_explicit_states_reconcile_only_changed_projects(tmp_path):
    local_db = LocalVersionKV(tmp_path / "local.db", tmp_path / "local.json")
    local_db.set_with_file_serial("popular", 1, PACKAGE_FILES_METADATA_ONLY)
    local_db.set_with_file_serial("other", 1, 1)
    other_path = write_simple_project(tmp_path, "other", "other.whl")
    other_path.parent.mkdir(parents=True, exist_ok=True)
    other_path.write_bytes(b"package")
    syncer = StateSync(
        tmp_path, local_db, {"popular": 1, "other": 1}, sync_packages=True
    )

    plan = syncer.determine_sync_plan(
        local_db.dump(skip_invalid=False),
        checker(),
        local_file_serials=local_db.dump_file_serials(skip_invalid=False),
    )
    assert plan.update == ["popular"]
    assert plan.package_remove == ["other"]

    assert syncer.do_sync_plan(plan, checker(), Mock()) is True
    assert not other_path.exists()
    assert local_db.dump_file_serials() == {
        "popular": 1,
        "other": PACKAGE_FILES_METADATA_ONLY,
    }


def test_no_sync_update_is_filled_by_a_later_package_sync(tmp_path):
    local_db = LocalVersionKV(tmp_path / "local.db", tmp_path / "local.json")
    local_db.set_with_file_serial("popular", 1, 1)
    no_file_sync = StateSync(tmp_path, local_db, {"popular": 2}, sync_packages=False)
    first_plan = no_file_sync.determine_sync_plan(
        local_db.dump(skip_invalid=False),
        checker(),
        local_file_serials=local_db.dump_file_serials(skip_invalid=False),
    )
    assert no_file_sync.do_sync_plan(first_plan, checker(), Mock()) is True
    assert local_db.dump() == {"popular": 2}
    assert local_db.dump_file_serials() == {"popular": PACKAGE_FILES_PENDING}

    with_file_sync = StateSync(tmp_path, local_db, {"popular": 2}, sync_packages=True)
    second_plan = with_file_sync.determine_sync_plan(
        local_db.dump(skip_invalid=False),
        checker(),
        local_file_serials=local_db.dump_file_serials(skip_invalid=False),
    )
    assert second_plan.update == ["popular"]


def test_pending_state_is_filled_without_ordered_filters(tmp_path):
    local_db = LocalVersionKV(tmp_path / "local.db", tmp_path / "local.json")
    local_db.set_with_file_serial("demo", 3, PACKAGE_FILES_PENDING)
    syncer = StateSync(tmp_path, local_db, {"demo": 3}, sync_packages=True)

    plan = syncer.determine_sync_plan(
        local_db.dump(skip_invalid=False),
        unfiltered_checker(),
        local_file_serials=local_db.dump_file_serials(skip_invalid=False),
    )

    assert plan.update == ["demo"]
