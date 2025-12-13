"""
Sanitize image metadata for staged images.

Features:
- Works on staged files (git diff --cached).
- Uses ExifTool to remove sensitive metadata (RAW + JPEG + TIFF + PNG etc.).
- Removes metadata based on a blacklist that spans EXIF, XMP, IPTC.
- Re-stages sanitized files into the index.

Usage:
- As script: python -m dev_tools.sanitize_images
- Called by pre-commit hook (defined in .pre-commit-config.yaml)
"""

from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# These are raw ExifTool args that get passed directly.
EXIFTOOL_REMOVE_ARGS = [
    # Geolocation (already have but keep full coverage)
    "-GPS:all=",
    "-EXIF:GPS*=",
    "-XMP:GPS*=",

    # Maker / device identifiers
    "-EXIF:SerialNumber=",
    "-EXIF:CameraSerialNumber=",
    "-EXIF:LensSerialNumber=",
    "-EXIF:CameraOwnerName=",
    "-EXIF:OwnerName=",
    "-MakerNotes:all=",           # vendor maker notes (removes vendor-specific identifiers)
    "-MakerNote:all=",

    # Identity / person / copyright / creator fields
    "-EXIF:Artist=",
    "-XMP:Creator=",
    "-XMP-dc:creator=",
    "-IPTC:By-line=",
    "-IPTC:By-lineTitle=",
    "-IPTC:Credit=",
    "-IPTC:CopyrightNotice=",
    "-IPTC:Source=",
    "-IPTC:Contact=",
    "-XMP:CreatorTool=",
    "-XMP:Rights=",

    # Unique IDs, UUIDs, and instance/document IDs
    "-EXIF:ImageUniqueID=",
    "-XMP:ImageUniqueID=",
    "-XMP-xmpMM:DocumentID=",
    "-XMP-xmpMM:InstanceID=",
    "-XMP-xmpMM:OriginalDocumentID=",

    # Timestamps & timezone-related fields
    "-EXIF:OffsetTime=",
    "-EXIF:OffsetTimeOriginal=",
    "-EXIF:OffsetTimeDigitized=",
    "-XMP:MetadataDate=",
    "-XMP:ModifyDate=",

    # Location text and IPTC location fields (city/country/sub-location)
    "-IPTC:City=",
    "-IPTC:Sub-location=",
    "-IPTC:Province-State=",
    "-IPTC:Country-PrimaryLocationName=",
    "-XMP:Location*=",

    # Windows / Microsoft XP / platform fields (often free-text)
    "-EXIF:XPAuthor=",
    "-EXIF:XPComment=",
    "-EXIF:XPTitle=",
    "-EXIF:XPKeywords=",
    "-EXIF:XPSubject=",

    # ICC / profile creators and descriptive strings
    "-ICC:ProfileCreator=",
    "-ICC:ProfileDescription=",

    # Software / editing history / history
    "-EXIF:Software=",
    "-XMP:History=",
    "-XMP:HistoryAction=",
    "-XMP:HistoryWhen=",
    "-XMP-photoshop:Instructions=",

    # IPTC fields that often contain photographer/contact names or captions
    "-IPTC:Creator=",
    "-IPTC:By-line=",
    "-IPTC:Caption-Abstract=",
    "-IPTC:Headline=",
    "-IPTC:Source=",

    # Misc identifiers and technical fields that can contain free-text
    "-DocumentName=",
    "-EXIF:UserComment=",
    "-XMP:UserComment=",
]

# Supported extensions (images + RAW formats)
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".tif", ".tiff", ".png",
    ".cr2", ".cr3", ".nef", ".arw", ".rw2",
    ".dng", ".orf", ".raf",
}


def _run_command_check(cmd, capture_output=True, check=True):
    """Run subprocess with basic error handling."""
    res = subprocess.run(cmd, capture_output=capture_output, text=True)
    if check and res.returncode != 0:
        raise subprocess.CalledProcessError(
            res.returncode, cmd, output=res.stdout, stderr=res.stderr
        )
    return res


def is_exiftool_available() -> bool:
    return shutil.which("exiftool") is not None


def sanitize_file(path: Path) -> None:
    """Sanitize a single file in place."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() not in IMAGE_EXTS:
        return  # Not an image; ignore

    if not is_exiftool_available():
        raise RuntimeError(
            "exiftool is not available. "
            "IMAGE FILES WERE NOT SANITIZED AND MAY CONTAIN EXIF METADATA."
        )

    cmd = (
        ["exiftool", "-quiet", "-ignoreMinorErrors", "-overwrite_original"]
        + EXIFTOOL_REMOVE_ARGS
        + ["--", str(path)]
    )
    _run_command_check(cmd)


def get_staged_images() -> List[Path]:
    """Return list of staged files that look like images."""
    try:
        out = subprocess.check_output(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            text=True,
        )
    except subprocess.CalledProcessError:
        return []

    files = [Path(s) for s in out.splitlines() if s]
    return [p for p in files if p.suffix.lower() in IMAGE_EXTS]


def readd_to_index(path: Path) -> None:
    """Re-stage the modified file to the git index."""
    subprocess.check_call(["git", "add", str(path)])


def sanitize_staged_images() -> int:
    """Main entry for pre-commit: sanitize staged images and re-stage them."""
    images = get_staged_images()
    count = 0

    for p in images:
        try:
            sanitize_file(p)
            readd_to_index(p)
            count += 1
        except Exception as e:
            print(f"ERROR: could not sanitize {p}: {e}", file=sys.stderr)
            raise  # Fail the commit

    return count


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint. If paths of files/directories are given, sanitize them; otherwise sanitize staged images."""
    argv = argv if argv is not None else sys.argv[1:]

    if argv:
        # Sanitize only the explicitly passed paths
        for a in argv:
            p = Path(a)
            if p.is_dir():
                # If it's a directory, process all files with valid extensions
                for file in p.rglob("*"):
                    if file.is_file() and file.suffix in IMAGE_EXTS:
                        sanitize_file(file)
                        readd_to_index(file)
            else:
                # If it's a file, process it directly
                sanitize_file(p)
                readd_to_index(p)
        return 0

    # Default: sanitize staged images
    sanitize_staged_images()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"Sanitization failed: {exc}", file=sys.stderr)
        sys.exit(1)
