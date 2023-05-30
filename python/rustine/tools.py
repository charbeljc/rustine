import logging
import os
import random
import re
from pathlib import Path
from typing import Optional, NewType
from pubgrub import Requirement, Pep508Error

MINIMUM_SUPPORTED_PYTHON_MINOR = 7

logger = logging.getLogger(__name__)
user_agent = "monotrail-resolve-prototype/0.0.1-dev1+cat <konstin@mailbox.org>"
base_dir = Path(__file__).parent.parent
default_cache_dir = base_dir.joinpath("cache")
normalizer = re.compile(r"[-_.]+")

NormalizedName = NewType("NormalizedName", str)


def normalize(name: str) -> NormalizedName:
    return NormalizedName(normalizer.sub("-", name).lower())


class Cache:
    """Quick and simple cache abstraction that can be turned off for the tests"""

    root_cache_dir: Path
    read: bool
    write: bool
    refresh_versions: bool

    def __init__(
        self,
        root_cache_dir: Path,
        read: bool = True,
        write: bool = True,
        refresh_versions: bool = False,
    ):
        self.root_cache_dir = root_cache_dir
        self.read = read
        self.write = write
        self.refresh_versions = refresh_versions

    def filename(self, bucket: str, name: str) -> Path:
        return self.root_cache_dir.joinpath(bucket).joinpath(name)

    def get_filename(self, bucket: str, name: str) -> Optional[Path]:
        """Middle abstraction for rust bridging"""
        if not self.read:
            return None

        return self.filename(bucket, name)

    def get(self, bucket: str, name: str) -> Optional[str]:
        if not self.read:
            return None

        filename = self.filename(bucket, name)
        # Avoid an expensive is_file call
        try:
            return filename.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    def set(self, bucket: str, name: str, content: str):
        if not self.write:
            return False
        filename = self.filename(bucket, name)
        filename.parent.mkdir(exist_ok=True, parents=True)
        # tempfile to avoid broken cache entry
        characters = "abcdefghijklmnopqrstuvwxyz0123456789_"
        temp_name = "".join(random.choices(characters, k=8))
        temp_file = filename.parent.joinpath(temp_name)
        temp_file.write_text(content, encoding="utf-8")
        os.replace(temp_file, filename)


def parse_requirement_fixup(
    requirement: str, debug_source: Optional[str]
) -> Requirement:
    """Fix unfortunately popular errors such as `elasticsearch-dsl (>=7.2.0<8.0.0)` in
    django-elasticsearch-dsl 7.2.2 with a regex heuristic

    None is a shabby way to signal not to warn, this should be solved properly by only
    caching once and then warn
    """
    try:
        return Requirement(requirement)
    except Pep508Error:
        try:
            # Add the missing comma
            requirement_parsed = Requirement(
                re.sub(r"(\d)([<>=~^!])", r"\1,\2", requirement)
            )
            if debug_source:
                logger.warning(
                    f"Requirement `{requirement}` for {debug_source} is invalid"
                    " (missing comma)"
                )
            return requirement_parsed
        except Pep508Error:
            pass
        # Didn't work with the fixup either? raise the error with the original string
        raise
