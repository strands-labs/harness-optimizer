"""Generate AppWorld task CSVs from the downloaded AppWorld data bundle.

The runtime loads tasks from ``<APPWORLD_DATASET_CSV>/<split>.csv`` (fast, one file
read at startup). Those CSVs are a flattened cache of AppWorld's own task data, which
``appworld download data`` unpacks from the authors' bundle
(``https://s3.us-west-2.amazonaws.com/appworld.dev/data-<version>.bundle``) into::

    <data>/datasets/<split>.txt        # task-id list per split
    <data>/tasks/<task_id>/specs.json  # {instruction, supervisor, datetime, ...}

This script reads those unpacked files directly (no ``AppWorld`` instantiation, so it's
fast and exits cleanly) and writes one CSV row per task, matching the schema
``dataset/dataset.py``'s ``AppWorldDataSample.from_dict`` expects. ``supervisor`` /
``app_descriptions`` / ``metadata`` are JSON-encoded cells (``load_from_csv`` decodes them).

Run at image-build time, after ``appworld download data`` (see the Dockerfiles).

Usage:
    python generate_csvs.py [OUT_DIR] [SPLIT ...]
    # default OUT_DIR=./datasets/appworld, default splits=train dev test_normal test_challenge
"""

import csv
import json
import os
import sys

FIELDS = ["id", "input", "task_id", "instruction", "supervisor", "app_descriptions", "metadata"]
DEFAULT_SPLITS = ["train", "dev", "test_normal", "test_challenge"]
REMOTE_APIS_PORT = int(os.getenv("APPWORLD_REMOTE_APIS_PORT", "9000"))


def _data_root() -> str:
    """Locate the unpacked AppWorld data dir (from path_store, else common fallbacks)."""
    try:
        from appworld.common.path_store import path_store
        if os.path.isdir(path_store.data):
            return path_store.data
    except Exception:
        pass
    for cand in (os.path.join(os.getcwd(), "data"), "/app/data"):
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(
        "Could not find the AppWorld data dir. Run `appworld download data` first, "
        "or set the working directory to where ./data lives."
    )


def _app_descriptions() -> dict:
    """The per-task app catalog (identical across tasks: all non-admin apps)."""
    from appworld.apps import get_all_apps, get_app_to_description
    app_to_description = get_app_to_description()
    allowed = get_all_apps(skip_admin=True)
    return {app: app_to_description[app] for app in allowed if app in app_to_description}


def _json_cell(value) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return value if value is not None else ""


def generate_split(split: str, out_dir: str, data_root: str, app_desc: dict) -> int:
    ids_file = os.path.join(data_root, "datasets", f"{split}.txt")
    if not os.path.isfile(ids_file):
        raise FileNotFoundError(f"Split id file not found: {ids_file}")
    task_ids = [line.strip() for line in open(ids_file) if line.strip()]

    rows = []
    for task_id in task_ids:
        spec_path = os.path.join(data_root, "tasks", task_id, "specs.json")
        spec = json.load(open(spec_path))
        rows.append({
            "id": task_id,
            "input": spec["instruction"],
            "task_id": task_id,
            "instruction": spec["instruction"],
            "supervisor": _json_cell(spec["supervisor"]),
            "app_descriptions": _json_cell(app_desc),
            "metadata": _json_cell({"remote_apis_port": REMOTE_APIS_PORT}),
        })

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{split}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} tasks -> {path}")
    return len(rows)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("datasets", "appworld")
    splits = sys.argv[2:] if len(sys.argv) > 2 else DEFAULT_SPLITS
    data_root = _data_root()
    app_desc = _app_descriptions()
    total = sum(generate_split(s, out_dir, data_root, app_desc) for s in splits)
    print(f"done: {total} tasks across {len(splits)} split(s) in {out_dir}")


if __name__ == "__main__":
    main()
