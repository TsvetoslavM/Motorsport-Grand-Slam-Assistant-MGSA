import csv
import json
import os
from typing import Iterable, List, Tuple, Union


Point = Tuple[float, float]


def _to_point(obj: Union[Iterable[float], dict]) -> Point:
    if isinstance(obj, dict):
        return float(obj["x"]), float(obj["y"])
    x, y = obj
    return float(x), float(y)


def load_points(file_path: str) -> List[Point]:
    """Load track points from CSV or JSON.

    CSV: rows as x,y
    JSON: list of [x,y] or list of {"x":..,"y":..}
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Points file not found: {file_path}")
    _, ext = os.path.splitext(file_path.lower())
    if ext == ".csv":
        points: List[Point] = []
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or row[0].strip().startswith("#"):
                    continue
                if len(row) < 2:
                    continue
                points.append((float(row[0]), float(row[1])))
        return points
    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [_to_point(item) for item in data]
    raise ValueError(f"Unsupported file type: {ext}")


def load_centerline_with_widths(file_path: str) -> List[Tuple[float, float, float, float]]:
    """Load CSV with columns: x,y,left_width,right_width.

    Lines starting with # are ignored. Extra columns are ignored.
    Returns list of tuples (x,y,left,right).
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    rows: List[Tuple[float, float, float, float]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or str(row[0]).strip().startswith("#"):
                continue
            if len(row) < 4:
                continue
            x = float(row[0]); y = float(row[1]); lw = float(row[2]); rw = float(row[3])
            rows.append((x, y, lw, rw))
    return rows



