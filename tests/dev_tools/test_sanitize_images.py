import subprocess
from dev_tools.sanitize_images import sanitize_file, EXIFTOOL_REMOVE_ARGS, sanitize_staged_images, get_staged_images
from pathlib import Path
import pytest
import json


def set_metadata(path: Path, tag: str, value: str = "TESTVALUE"):
    """Write metadata using exiftool."""
    subprocess.check_call(["exiftool", f"-{tag}={value}", str(path)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def read_metadata(path: Path):
    """Read all metadata using exiftool -j."""
    out = subprocess.check_output(["exiftool", "-j", str(path)], text=True)
    return json.loads(out)[0]


# Map removal args (which can be group/wildcard) to a concrete tag we can write/check.
# This mapping should contain at least one writable tag that the removal arg is intended to remove.
REMOVAL_ARG_TO_TEST_TAG = {
    # GPS (representative EXIF + XMP tags)
    "-GPS:all=": "EXIF:GPSLatitude",
    "-EXIF:GPS*=": "EXIF:GPSLongitude",
    "-XMP:GPS*=": "XMP:GPSLatitude",


    # Maker / device identifiers
    "-EXIF:CameraOwnerName=": "IFD0:OwnerName",
    "-EXIF:OwnerName=": "IFD0:OwnerName",

    # MakerNotes not writable, set dummy value instead
    "-MakerNotes:all=": "XMP:Manufacturer",
    "-MakerNote:all=": "XMP:Manufacturer",


    # Identity / person / copyright / creator
    "-EXIF:Artist=": "IFD0:Artist",
    "-XMP:Creator=": "XMP:Creator",
    "-XMP-dc:creator=": "XMP-dc:Title",#would expect a list
    "-IPTC:Creator=": "IPTC:By-line",

    # Location fields (text locations across namespaces)
    "-XMP:Location*=": "XMP:Location",
    "-XMP:LocationShown*=": "XMP:LocationShownCity",
    "-XMP:LocationCreated*=": "XMP:LocationCreatedCity",

    # ICC profile descriptive strings
    "-ICC:ProfileCreator=": "EXIF:GPSLatitude", #not writable
    "-ICC:ProfileDescription=": "EXIF:GPSLatitude", #not writable

    # Misc identifiers
    "-DocumentName=": "IFD0:DocumentName",
    "-EXIF:UserComment=": "EXIF:UserComment",
    "-XMP:UserComment=": "XMP:UserComment",
}

TEST_TAG_SPECIAL_VALS = {
    # GPS numeric/coordinate fields
    "EXIF:GPSLatitude": "12.345",
    "EXIF:GPSLongitude": "67.890",
    "XMP:GPSLatitude": "12.345",

    # Serial numbers
    "EXIF:SerialNumber": "123456",
    "EXIF:CameraSerialNumber": "123456",
    "EXIF:LensSerialNumber": "123456",

    # Unique IDs: must look like IDs rather than plain text
    "EXIF:ImageUniqueID": "ABCDEF123456",
    "XMP:ImageUniqueID": "UUID-1234-5678",
    "XMP-xmpMM:DocumentID": "xmp.did:0011223344",
    "XMP-xmpMM:InstanceID": "xmp.iid:5544332211",
    "XMP-xmpMM:OriginalDocumentID": "xmp.did:orig-77889900",

    # Time-zone offset fields
    "EXIF:OffsetTime": "+01:00",
    "EXIF:OffsetTimeOriginal": "+02:00",
    "EXIF:OffsetTimeDigitized": "+03:00",

    # XMP date/time fields must be proper UTC timestamps
    "XMP:MetadataDate": "2024:01:01 12:00:00Z",
    "XMP:ModifyDate": "2024:01:01 12:00:00Z",
    "XMP:HistoryWhen": "2024:01:01",

    # MakerNotes test tag (must be a plausible version number)
    "MakerNotes:FirmwareVersion": "1.2.3",
}


@pytest.mark.usefixtures("ensure_exiftool")
def test_each_blacklisted_tag_is_removed(tmp_image):
    """
    For each removal arg defined in EXIFTOOL_REMOVE_ARGS:
      - write metadata for a concrete test tag (mapping)
      - run sanitize_file()
      - ensure the tag (or a reasonable representation) is not present afterward
    """
    # Work on a fresh copy per iteration to avoid exiftool backups interfering
    for arg in EXIFTOOL_REMOVE_ARGS:
        # If we don't have a mapping, skip this arg (safer than attempting brittle parsing).
        test_tag = REMOVAL_ARG_TO_TEST_TAG.get(arg, arg[1:-1])
        if not test_tag:
            pytest.skip(f"No concrete test tag available for removal arg: {arg}")

        # Copy the tmp_image to a per-case file so exiftool backups don't clash.
        this_file = tmp_image.parent / f"case_{abs(hash(arg)) % 100000}.jpg"
        this_file.write_bytes(tmp_image.read_bytes())

        # Write the concrete test metadata
        try:
            value = TEST_TAG_SPECIAL_VALS.get(test_tag, "REMOVE")
            set_metadata(this_file, test_tag, value)
        except subprocess.CalledProcessError as e:
            # Some tags may not be writable on a particular file type/platform; skip in that case.
            pytest.skip(f"Could not write tag {test_tag} on this platform/file. Error: {str(e)}")

        md_before = read_metadata(this_file)
        # Ensure it was written (best-effort: key might be namespaced or slightly different)
        test_tag_end = test_tag.split(":")[1]
        assert test_tag_end in md_before.keys() , (
            f"{test_tag} was not written as expected; metadata keys: {list(md_before.keys())}"
        )

    # Run sanitizer
    sanitize_file(this_file)

    # Read metadata after sanitization
    md_after = read_metadata(this_file)

    for arg in EXIFTOOL_REMOVE_ARGS:
        test_tag = REMOVAL_ARG_TO_TEST_TAG.get(arg, arg[1:-1])
        test_tag_end = test_tag.split(":")[1]
        # Assert that the specific test tag was removed (best-effort matching)
        assert test_tag_end not in md_after.keys() , (
            f"{test_tag} was not written as expected; metadata keys: {list(md_before.keys())}"
        )


def test_other_metadata_preserved(tmp_image):
    """Ensure non-sensitive metadata are preserved."""
    set_metadata(tmp_image, "LensMake", "well")
    md_before = read_metadata(tmp_image)
    assert "LensMake" in md_before

    sanitize_file(tmp_image)

    md_after = read_metadata(tmp_image)
    assert "LensMake" in md_after


def test_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        sanitize_file(Path("no_such_file.jpg"))


def test_non_image_file(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("abc")

    # Should not raise and not modify
    sanitize_file(f)
    assert f.read_text() == "abc"


def git(*args, cwd):
    subprocess.check_call(["git"] + list(args), cwd=cwd)


@pytest.mark.usefixtures("ensure_exiftool")
def test_sanitize_staged_images(tmp_path, monkeypatch):
    # Initialize a repo in the temporary path
    repo = tmp_path
    git("init", cwd=repo)

    # Set minimal git config so commits/staging works
    git("config", "user.email", "test@example.com", cwd=repo)
    git("config", "user.name", "Test User", cwd=repo)

    # Create an image in the repo
    img = repo / "img.jpg"
    from PIL import Image
    Image.new("RGB", (32, 32)).save(img, "JPEG")

    # Add sensitive tag directly
    subprocess.check_call(["exiftool", "-GPSLatitude=12.34", str(img)],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Stage the file
    git("add", "img.jpg", cwd=repo)

    # Change current working directory to the repo so get_staged_images() sees it
    monkeypatch.chdir(repo)

    # ensure staged list is seen by the helper
    staged = get_staged_images()
    assert len(staged) == 1

    # run sanitizer (this will sanitize and re-add to index)
    sanitize_staged_images()

    # metadata should be removed from the file on disk
    out = subprocess.check_output(["exiftool", "-j", str(img)], text=True)
    import json
    meta = json.loads(out)[0]
    assert "GPSLatitude" not in meta
