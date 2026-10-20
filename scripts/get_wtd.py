"""Step 1: put the WTD files in the team's standard places (same names on every computer).

    python scripts/get_wtd.py --list               # what is inside the 24.2 GB zip -> data/raw/wtd/zip_contents.txt
    python scripts/get_wtd.py --csv                # every CSV/XLSX (transcripts, CPS codes) -> data/raw/wtd/csv/
    python scripts/get_wtd.py --videos 1 2 3       # the video named in configs/wtd_videos.csv -> data/raw/wtd/video/g01.mp4 ...

Only the needed files are downloaded (remotezip reads parts of the zip on Zenodo: pip install remotezip).
If that fails, download the whole zip once in a browser and add  --zip <path to the zip>  to any command.

configs/wtd_videos.csv (written once by the Data owner, committed to git) says which of a group's videos we use:
    group,zip_path,view
    1,<path exactly as in zip_contents.txt>,<e.g. front: all three faces visible>
Standard places afterwards
    data/raw/wtd/csv/Group_01_Oracle.csv, ..., All_Groups_CPS.csv
    data/raw/wtd/video/g01.mp4, g02.mp4, g03.mp4
"""
from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from ie4422.config import repo_path

URL = "https://zenodo.org/records/10252341/files/Weights%20Task%20Dataset.zip?download=1"


def open_zip(local: str | None) -> zipfile.ZipFile:
    if local:
        return zipfile.ZipFile(local)
    from remotezip import RemoteZip  # a ZipFile that downloads only the parts it reads
    return RemoteZip(URL)


def extract(z: zipfile.ZipFile, member: str, dest: Path) -> None:
    info = z.getinfo(member)
    if dest.exists() and dest.stat().st_size == info.file_size:
        print(f"  already there: {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with z.open(member) as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out, 1 << 20)
    print(f"  {member}  ->  {dest}  ({info.file_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--csv", action="store_true")
    ap.add_argument("--videos", type=int, nargs="*", metavar="GROUP")
    ap.add_argument("--zip", help="local copy of the zip instead of Zenodo")
    a = ap.parse_args()
    if not (a.list or a.csv or a.videos):
        ap.error("choose --list, --csv and/or --videos")

    root = repo_path("data/raw/wtd")
    with open_zip(a.zip) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        if a.list:
            root.mkdir(parents=True, exist_ok=True)
            lines = [f"{i.file_size / 1e6:10.1f} MB  {i.filename}" for i in infos]
            (root / "zip_contents.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"{len(infos)} files -> {root / 'zip_contents.txt'}")
            print("videos (pick the view showing all three students for configs/wtd_videos.csv):")
            for line in lines:
                if line.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    print(line)

        if a.csv:
            tables = [i.filename for i in infos if i.filename.lower().endswith((".csv", ".xlsx"))]
            names = [PurePosixPath(t).name for t in tables]
            for t, name in zip(tables, names):
                if names.count(name) > 1:  # same file name in two folders: keep the folder in the name
                    name = f"{PurePosixPath(t).parent.name}__{name}"
                extract(z, t, root / "csv" / name)
            print(f"{len(tables)} tables in {root / 'csv'}")

        if a.videos:
            cfg = repo_path("configs/wtd_videos.csv")
            if not cfg.exists():
                raise SystemExit("configs/wtd_videos.csv is missing: the Data owner writes it after --list")
            with open(cfg, encoding="utf-8") as f:
                chosen = {int(r["group"]): r["zip_path"].strip() for r in csv.DictReader(f)}
            for g in a.videos:
                if g not in chosen:
                    raise SystemExit(f"group {g} is not in configs/wtd_videos.csv")
                extract(z, chosen[g], root / "video" / f"g{g:02d}.mp4")


if __name__ == "__main__":
    main()
