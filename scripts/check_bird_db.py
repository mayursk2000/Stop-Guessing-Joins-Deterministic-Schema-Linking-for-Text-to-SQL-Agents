"""Inspect one BIRD SQLite database and its dev question count.

Example:
    python scripts/check_bird_db.py --bird-dev /content/bird/dev_20240627 --db toxicology
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sqlite3


def _size_mb(path: pathlib.Path) -> float:
    return path.stat().st_size / 1_000_000


def _find_dev_dir(path: pathlib.Path) -> pathlib.Path:
    if (path / "dev.json").exists():
        return path
    matches = list(path.rglob("dev.json"))
    if not matches:
        raise FileNotFoundError(f"dev.json not found under {path}")
    return matches[0].parent


def _find_db_root(dev_dir: pathlib.Path) -> pathlib.Path:
    direct = dev_dir / "dev_databases"
    if direct.exists():
        return direct
    matches = [p for p in dev_dir.rglob("dev_databases") if p.is_dir()]
    if not matches:
        raise FileNotFoundError(f"dev_databases not found under {dev_dir}")
    return matches[0]


def inspect_bird_db(bird_dev: pathlib.Path, db_id: str) -> None:
    dev_dir = _find_dev_dir(bird_dev)
    db_root = _find_db_root(dev_dir)
    db_dir = db_root / db_id
    sqlite_path = db_dir / f"{db_id}.sqlite"

    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite DB not found: {sqlite_path}")

    dev = json.loads((dev_dir / "dev.json").read_text(encoding="utf-8"))
    question_count = sum(1 for q in dev if q.get("db_id") == db_id)

    total_files = sum(p.stat().st_size for p in db_dir.rglob("*") if p.is_file())
    print(f"db_id: {db_id}")
    print(f"dev_dir: {dev_dir}")
    print(f"db_dir: {db_dir}")
    print(f"sqlite: {sqlite_path}")
    print(f"db_files_mb: {total_files / 1_000_000:.1f}")
    print(f"sqlite_mb: {_size_mb(sqlite_path):.1f}")
    print(f"dev_questions: {question_count}")

    conn = sqlite3.connect(sqlite_path)
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
        ]
        print(f"tables: {len(tables)}")

        counts: list[tuple[str, int]] = []
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            count = int(conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0])
            counts.append((table, count))

        total_rows = sum(count for _, count in counts)
        print(f"total_rows: {total_rows}")
        print("table_counts:")
        for table, count in sorted(counts, key=lambda item: (-item[1], item[0])):
            print(f"  {table}: {count}")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bird-dev",
        type=pathlib.Path,
        default=pathlib.Path("/content/bird"),
        help="BIRD dev directory or a parent directory containing dev.json.",
    )
    parser.add_argument("--db", default="toxicology", help="BIRD db_id to inspect.")
    args = parser.parse_args()
    inspect_bird_db(args.bird_dev, args.db)


if __name__ == "__main__":
    main()
