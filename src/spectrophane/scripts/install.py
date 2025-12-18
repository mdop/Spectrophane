import os
import sys
import pathlib
import urllib.request

from spectrophane.io.resources import USER_DATA_DIR

# List of CC BY-SA CIE CSV resources you want to download
CIE_CSV_URLS = [
    "https://files.cie.co.at/CIE_std_illum_D65.csv",
    "https://files.cie.co.at/CIE_xyz_1931_2deg.csv",
    "https://files.cie.co.at/CIE_xyz_1964_10deg.csv"
]


def download_file(url: str, dest: pathlib.Path) -> None:
    """Download a single file with a simple progress message to a specified directory. Skips if file already exists."""
    if dest.exists():
        print(f"[SKIP] {dest.name} already exists.")
        return

    print(f"[DOWNLOAD] {url} → {dest}")

    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
            dest.write_bytes(data)
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        return

    print(f"[OK] Saved {dest}")


def main():
    base_dir = USER_DATA_DIR / "CIE"
    base_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading CIE standard data (CC BY-SA 4.0, © CIE)")
    print(f"Downloading to directory: {base_dir}")

    for url in CIE_CSV_URLS:
        filename = url.split("/")[-1]
        dest = base_dir / filename
        download_file(url, dest)

    print("Done.")


if __name__ == "__main__":
    main()
