import json
import sqlite3
from pathlib import Path

from .constants import PACKAGE_NOT_FOUND_SERIAL
from .filesystem import overwrite


class LocalVersionKV:
    """
    A key-value database wrapper over sqlite3.

    As it would have consistency issue if it's writing while downstream is downloading the database.
    An extra "jsonpath" is used, to store kv results when necessary.
    """

    def __init__(self, dbpath: Path, jsonpath: Path) -> None:
        self.conn = sqlite3.connect(dbpath)
        self.jsonpath = jsonpath
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS local("
            "key TEXT PRIMARY KEY, value INT NOT NULL, file_serial INT)"
        )
        columns = {row[1] for row in cur.execute("PRAGMA table_info(local)")}
        if "file_serial" not in columns:
            cur.execute("ALTER TABLE local ADD COLUMN file_serial INT")
        self.conn.commit()

    def get(self, key: str) -> int | None:
        cur = self.conn.cursor()
        res = cur.execute("SELECT value FROM local WHERE key = ?", (key,))
        row = res.fetchone()
        return row[0] if row else None

    INSERT_SQL = "INSERT INTO local (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value"

    def set(self, key: str, value: int) -> None:
        cur = self.conn.cursor()
        cur.execute(self.INSERT_SQL, (key, value))
        self.conn.commit()

    SET_WITH_FILE_SERIAL_SQL = (
        "INSERT INTO local (key, value, file_serial) VALUES (?, ?, ?) "
        "ON CONFLICT(key) DO UPDATE SET "
        "value=excluded.value, file_serial=excluded.file_serial"
    )

    def set_with_file_serial(
        self, key: str, value: int, file_serial: int | None
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(self.SET_WITH_FILE_SERIAL_SQL, (key, value, file_serial))
        self.conn.commit()

    def set_file_serial(self, key: str, file_serial: int | None) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE local SET file_serial = ? WHERE key = ?", (file_serial, key)
        )
        self.conn.commit()

    def batch_set_file_serials(self, values: dict[str, int]) -> None:
        cur = self.conn.cursor()
        cur.executemany(
            "UPDATE local SET file_serial = ? WHERE key = ?",
            [(value, key) for key, value in values.items()],
        )
        self.conn.commit()

    def batch_set(self, d: dict[str, int]) -> None:
        cur = self.conn.cursor()
        kvs = list(d.items())
        cur.executemany(self.INSERT_SQL, kvs)
        self.conn.commit()

    def remove(self, key: str) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM local WHERE key = ?", (key,))
        self.conn.commit()

    def remove_invalid(self) -> int:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM local WHERE value = ?", (PACKAGE_NOT_FOUND_SERIAL,))
        rowcnt = cur.rowcount
        self.conn.commit()
        return rowcnt

    def nuke(self, commit: bool = True) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM local")
        if commit:
            self.conn.commit()

    def keys(self, skip_invalid: bool = True) -> list[str]:
        cur = self.conn.cursor()
        if skip_invalid:
            res = cur.execute(
                "SELECT key FROM local WHERE value != ?",
                (PACKAGE_NOT_FOUND_SERIAL,),
            )
        else:
            res = cur.execute("SELECT key FROM local")
        rows = res.fetchall()
        return [row[0] for row in rows]

    def dump(self, skip_invalid: bool = True) -> dict[str, int]:
        cur = self.conn.cursor()
        if skip_invalid:
            res = cur.execute(
                "SELECT key, value FROM local WHERE value != ?",
                (PACKAGE_NOT_FOUND_SERIAL,),
            )
        else:
            res = cur.execute("SELECT key, value FROM local")
        rows = res.fetchall()
        return {row[0]: row[1] for row in rows}

    def dump_file_serials(self, skip_invalid: bool = True) -> dict[str, int | None]:
        cur = self.conn.cursor()
        if skip_invalid:
            res = cur.execute(
                "SELECT key, file_serial FROM local WHERE value != ?",
                (PACKAGE_NOT_FOUND_SERIAL,),
            )
        else:
            res = cur.execute("SELECT key, file_serial FROM local")
        rows = res.fetchall()
        return {row[0]: row[1] for row in rows}

    def dump_json(self, skip_invalid: bool = True) -> None:
        res = self.dump(skip_invalid)
        with overwrite(self.jsonpath) as f:
            json.dump(res, f, indent=2)
