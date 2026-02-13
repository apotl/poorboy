import json
import os
import sqlite3
import logging

logger = logging.getLogger("poorboy")


class TickerCache:
    """SQLite-backed ticker cache with dict-like interface.

    Supports multiprocessing via pickle: each worker lazily creates its own
    sqlite3.Connection. WAL mode + busy_timeout allow concurrent writes.
    """

    def __init__(self, db_path="ticker_cache.db"):
        self.db_path = db_path
        self._conn = None
        self._ensure_table()

    def _get_conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    def _ensure_table(self):
        conn = self._get_conn()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS tickers "
            "(symbol TEXT PRIMARY KEY, data TEXT)"
        )
        conn.commit()

    # -- Pickle support for multiprocessing --

    def __getstate__(self):
        return {"db_path": self.db_path}

    def __setstate__(self, state):
        self.db_path = state["db_path"]
        self._conn = None

    # -- Dict-like interface --

    def __getitem__(self, symbol):
        row = self._get_conn().execute(
            "SELECT data FROM tickers WHERE symbol = ?", (symbol,)
        ).fetchone()
        if row is None:
            raise KeyError(symbol)
        return json.loads(row[0])

    def __setitem__(self, symbol, value):
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO tickers (symbol, data) VALUES (?, ?)",
            (symbol, json.dumps(value)),
        )
        conn.commit()

    def get(self, symbol, default=None):
        row = self._get_conn().execute(
            "SELECT data FROM tickers WHERE symbol = ?", (symbol,)
        ).fetchone()
        if row is None:
            return default
        return json.loads(row[0])

    def __contains__(self, symbol):
        row = self._get_conn().execute(
            "SELECT 1 FROM tickers WHERE symbol = ?", (symbol,)
        ).fetchone()
        return row is not None

    def keys(self):
        rows = self._get_conn().execute("SELECT symbol FROM tickers").fetchall()
        return [r[0] for r in rows]

    def items(self):
        rows = self._get_conn().execute(
            "SELECT symbol, data FROM tickers"
        ).fetchall()
        return [(r[0], json.loads(r[1])) for r in rows]

    # -- Bulk operations --

    def bulk_load(self):
        """Return all rows as a plain dict."""
        rows = self._get_conn().execute(
            "SELECT symbol, data FROM tickers"
        ).fetchall()
        return {r[0]: json.loads(r[1]) for r in rows}

    def bulk_write(self, data):
        """Upsert all entries in a single transaction."""
        conn = self._get_conn()
        conn.executemany(
            "INSERT OR REPLACE INTO tickers (symbol, data) VALUES (?, ?)",
            [(symbol, json.dumps(value)) for symbol, value in data.items()],
        )
        conn.commit()

    def clear(self):
        """Delete all rows."""
        conn = self._get_conn()
        conn.execute("DELETE FROM tickers")
        conn.commit()

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # -- Migration --

    @classmethod
    def migrate_from_json(cls, json_path, db_path="ticker_cache.db"):
        """One-time migration: read JSON, bulk-insert into SQLite, rename JSON to .bak."""
        logger.info("Migrating ticker cache from %s to %s", json_path, db_path)
        with open(json_path) as f:
            data = json.loads(f.read())
        cache = cls(db_path)
        cache.bulk_write(data)
        cache.close()
        bak_path = json_path + ".bak"
        os.rename(json_path, bak_path)
        logger.info("Migration complete. JSON renamed to %s", bak_path)
