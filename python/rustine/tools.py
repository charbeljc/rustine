import re
from pathlib import Path
from typing import BinaryIO, NewType, Optional, Union

import requests
from logzero import logger
from pubgrub import Pep508Error, Requirement, Version, VersionSpecifiers

MINIMUM_SUPPORTED_PYTHON_MINOR = 7

user_agent = "monotrail-resolve-prototype/0.0.1-dev1+cat <konstin@mailbox.org>"
base_dir = Path(__file__).parent.parent
default_cache_dir = base_dir.joinpath("cache")
normalizer = re.compile(r"[-_.]+")

NormalizedName = NewType("NormalizedName", str)


def normalize(name: str) -> NormalizedName:
    return NormalizedName(normalizer.sub("-", name).lower())


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


class RemoteZipFile(BinaryIO):
    """Pretend local zip file that is actually querying the pypi for the exact ranges of
    the file. Requirement is that the server supports range requests
    (https://developer.mozilla.org/en-US/docs/Web/HTTP/Range_requests)

    Only implements the methods actually called by zipfile for what we do, we're lying
    about the type here

    Adapted to requests from konstin/monotrail-resolve
    """

    url: str
    pos: int
    len: int
    user_agent = user_agent

    def __init__(self, client: requests.Session, url: str):
        self.url = url
        self.pos = 0
        self.client = client

        response = self.client.head(self.url, headers={"user-agent": self.user_agent})
        response.raise_for_status()
        accept_ranges = response.headers.get("accept-ranges")
        assert accept_ranges == "bytes", (
            f"The server needs to `accept-ranges: bytes`, "
            f"but it says {accept_ranges} for {url}"
        )
        self.len = int(response.headers["content-length"])

    def seekable(self):
        return True

    def seek(self, offset: int, whence: int = 0):
        logger.debug("seek offset: %s, whence: %s", offset, whence)
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.len + offset
        else:
            raise ValueError(f"whence must be 0, 1 or 2 but it's {whence}")
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size: Optional[int] = None):
        # Here we could also use an end-open range, but we already have the information,
        # so let's keep track locally (which we when in doubt we can trust over the
        # server)
        logger.debug("read size: %s", size)
        if size:
            read_len = size
        else:
            read_len = self.len - self.pos
        # HTTP Ranges are zero-indexed and inclusive
        # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Range
        # https://developer.mozilla.org/en-US/docs/Web/HTTP/Range_requests
        headers = {
            "Range": f"bytes={self.pos}-{self.pos + read_len - 1}",
            "user-agent": self.user_agent,
        }
        response = self.client.get(self.url, headers=headers)
        data = response.content
        self.pos += read_len
        return data


def fixup_requirement(str_or_req: Union[str,Requirement]) -> Requirement:
    if type(str_or_req) is str:
        req = Requirement(str_or_req)
    else:
        req = str_or_req
    return normalize_requirement(req)


def normalize_requirement(req):
    marker: Optional[str] = req.marker
    if marker:
        marker = normalize_python_specifier(marker)

        sreq, _ = str(req).split(";")
        sreq = sreq.strip()
        req = Requirement(f"{sreq}; {marker}")
        # print(f"XXX: {req}")
    return req


def normalize_python_specifier(marker: str):
    # FIXME: very approximate
    if " in " in marker:
        connector = " or "
        operator = "=="
        symbol, versions = marker.split(" in ")
        if symbol.endswith(" not"):
            connector = " and "
            operator = "!="
        assert versions[0] in ("'", '"')
        assert versions[-1] in ("'", '"')
        delim = versions[0]
        versions = versions[1:-1].split()
        marker = connector.join(
            f"{symbol} {operator} {delim}{version}{delim}" for version in versions
        )
    assert "in " not in marker, marker
    return f"{marker}"


def version_matches(python_version: Version, requires_python: Optional[VersionSpecifiers]):
    matched = True
    if isinstance(requires_python, str):
        raise TypeError("got a string")
    if requires_python:
        if not all(vs.contains(python_version) for vs in requires_python):
            matched = False
    if not matched:
        logger.debug(
            "version-matches: %r, %r -> %s", requires_python, python_version, matched
        )
    return matched
