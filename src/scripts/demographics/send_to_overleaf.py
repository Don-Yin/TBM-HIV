from dirsync import sync
from pathlib import Path


if __name__ == "__main__":
    source_folder = Path("results", "images")
    target_folder = Path("/Users/donyin/Dropbox/Apps/Overleaf/report-individual-project-suggested/images")
    target_folder.mkdir(parents=True, exist_ok=True)
    sync(source_folder, target_folder, "sync")
