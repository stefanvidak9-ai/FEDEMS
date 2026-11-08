"""Inventory COFACTOR building metadata without copying or changing raw files."""
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "Dataset"
OUTPUT = ROOT / "docs/phase2/dataset_inventory.csv"
BUILDINGS = range(6479, 6496)


def parse_metadata(path):
    result = {}
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            if line.split(";", 1)[0].strip().strip('"') == "TimeStamp":
                break
            if ";" in line:
                key, value = line.rstrip("\n").split(";", 1)
                result[key.strip()] = value.strip()
    return result


def main():
    rows = []
    for bid in BUILDINGS:
        file = RAW / f"building_{bid}.txt"
        if not file.is_file():
            raise FileNotFoundError(file)
        entry = {"building_id": bid, "raw_filename": file.name}
        entry.update(parse_metadata(file))
        rows.append(entry)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} building inventory records: {OUTPUT}")


if __name__ == "__main__":
    main()
