"""
Download public HPO data files used by the pipeline.
Files fetched:
  - hp.obo             HPO ontology (~10 MB)
Released under CC-BY-4.0 by the HPO consortium.
"""
import os
import sys
import urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)
FILES = [
    ("hp.obo",
     "http://purl.obolibrary.org/obo/hp.obo"),
]
def download(name: str, url: str) -> None:
    dest = os.path.join(DATA_DIR, name)
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"✓ {name} already present ({size_mb:.1f} MB) — skipping")
        return
    print(f"↓ Downloading {name} from {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  saved to {dest} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  ✗ Failed: {e}", file=sys.stderr)
        sys.exit(1)
if __name__ == "__main__":
    print(f"Target directory: {DATA_DIR}\n")
    for name, url in FILES:
        download(name, url)
    print("\nAll public data files are ready.")
