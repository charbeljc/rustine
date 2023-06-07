import io
import tarfile
from pathlib import Path
from zipfile import ZipFile

import requests
from diskcache import Cache as DCache
import logging
from logzero import setup_logger
from platformdirs import AppDirs
from pubgrub import MarkerEnvironment, Requirement, Version, VersionSpecifiers
from requests import Response
from rustine.tools import normalize_python_specifier
from wheel_filename import ParsedWheelFilename, parse_wheel_filename

from rustine.tools import normalize

from .tools import RemoteZipFile

USER_AGENT = "rustine/0.0.1-dev0+meow <charbeljacquin@gmail.com>"

CACHE_DIR = Path(AppDirs().user_cache_dir).joinpath("rustine")
DEFAULT_CACHE = DCache(CACHE_DIR)

PYPI_BASE = "https://pypi.org"

PYPI_SIMPLE_URL = "https://pypi.org/simple"
PYPI_API_URL = "https://pypi.org/pypi"

DEVPI_BASE = "http://localhost:3141/root/pypi"

ARCH_TAGS = [
    "x86_64",
    "i686",
    "aarch64",
    "universal2",
    "arm64",
    "s390x",
    "ppc64le",
    "amd64",
    "armv7l",
    "ppc64",
]

PLATFORM_TAGS = {
    "linux": "Linux",
    "manylinux1": "Linux",
    "manylinux2014": "Linux",
    "manylinux_2_5": "Linux",
    "manylinux_2_12": "Linux",
    "manylinux_2_17": "Linux",
    "manylinux2010": "Linux",
    "musllinux_1_1": "Linux",
    "macosx_10_9": "Darwin",
    "macosx_10_10": "Darwin",
    "macosx_10_11": "Darwin",
    "macosx_10_12": "Darwin",
    "macosx_11_0": "Darwin",
    "macosx_10_11": "Darwin",
    "macosx_10_14": "Darwin",
    "macosx_10_15": "Darwin",
    "macosx_12_0": "Darwin",
    "win": "Windows",
    "win32": "Windows",
    "any": None,
}
logger = setup_logger(name="index")
logger.setLevel(logging.INFO)


def pltag2tuple(tag):
    arch = None
    for probe in ARCH_TAGS:
        if tag.endswith(f"_{probe}"):
            tag = tag.removesuffix(f"_{probe}")
            arch = probe
            break

    return PLATFORM_TAGS[tag], arch


def pytag2tuple(tag):
    if tag.startswith("cp"):
        major = tag[2]
        minor = tag[3:]
        if not minor:
            return "cypthon", Version(major)
        return "cpython", Version(f"{major}.{minor}")
    elif tag.startswith("pp"):
        major = tag[2]
        minor = tag[3:]
        if not minor:
            return "pypy", Version(major)
        return "pypy", Version(f"{major}.{minor}")
    elif tag.startswith("py"):
        major = tag[2]
        minor = tag[3:]
        if not minor:
            return None, Version(major)
        return None, Version(f"{major}.{minor}")
    else:
        raise NotImplementedError(f"tag2tuple for {tag}")


def wheel_tags_to_requirements(name, wheel_filename: ParsedWheelFilename):
    abi_tags = wheel_filename.abi_tags
    python_tags = wheel_filename.python_tags
    platform_tags = wheel_filename.platform_tags

    assert len(abi_tags) == 1
    # assert len(python_tags) == 1
    # assert len(platform_tags) == 1
    if len(platform_tags) != 1:
        # logger.warning("package: %s, wheel: %s", name, wheel_filename)
        pass
    python_tags = [pytag2tuple(tag) for tag in python_tags]
    python_expr = []
    for py_impl, py_version in python_tags:
        expr = []
        if py_impl:
            expr.append(f"implementation_name == '{py_impl}'")
        if py_version:
            if len(py_version.release) == 1:
                next = int(py_version.release[0]) + 1
                expr.append(
                    f"python_version >='{py_version}' and python_version < '{next}'"
                )
            elif len(py_version.release) == 2:
                expr.append(f"python_version ~= '{py_version}'")
            else:
                raise ValueError("invalid python")
        expr = " and ".join(expr)
        python_expr.append(expr)
    python_expr = " or ".join(python_expr)
    if python_expr:
        python_expr = f"({python_expr})"
    pl_name, pl_arch = pltag2tuple(platform_tags[0])

    markers = []
    if python_expr:
        markers.append(python_expr)

    if pl_name:
        markers.append(f"platform_system == '{pl_name}'")
    if pl_arch:
        markers.append(f"platform_machine == '{pl_arch}'")
    markers = " and ".join(markers)
    return Requirement(f"{name}; {markers}")


def fetch_wheel_metadata(name, version, filename, url):
    metadata_path = f"{name}-{version}.dist-info/METADATA"
    with requests.Session() as client:
        zipfile = ZipFile(RemoteZipFile(client, url))
        try:
            metadata_bytes = zipfile.read(metadata_path)
        except KeyError:
            metadata_bytes = None
            for zipped_file in zipfile.namelist():
                # TODO: Check that there's actually exactly one dist info directory
                #       and METADATA file
                if zipped_file.count("/") == 1 and zipped_file.endswith(
                    ".dist-info/METADATA"
                ):
                    metadata_bytes = zipfile.read(zipped_file)
                    break
            if not metadata_bytes:
                raise RuntimeError(
                    f"Missing METADATA file for {name} {version} {filename} {url}"
                ) from None
    return metadata_bytes


def fetch_sdist_metadata(name, version, filename, url):
    logger.debug("fetch_sdist_metadata: %s, %s, %s, %s", name, version, filename, url)
    resp = requests.get(url)
    tar = tarfile.open(mode="r:gz", fileobj=io.BytesIO(resp.content))
    metadatas = [name for name in tar.getnames() if "PKG-INFO" in name]
    with tar.extractfile(metadatas[0]) as f:
        metadata_bytes = f.read()
    return metadata_bytes


def fetch_json(url, etag=None) -> Response:
    logger.debug("fetch-json: %s, %s", url, etag)
    headers = {
        "user-agent": USER_AGENT,
        "accept": "application/vnd.pypi.simple.v1+json",
    }
    if etag:
        headers["if-none-match"] = f'"{etag}"'
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    # logger.debug("get %s, status-code: %s", url, resp.status_code)
    # for h in resp.headers:
    #     logger.debug("%s: %s", h, resp.headers[h])
    return resp


def fetch_headers(url) -> Response:
    headers = {
        "user-agent": USER_AGENT,
        "accept": "application/vnd.pypi.simple.v1+json",
    }
    resp = requests.head(url, headers=headers)
    resp.raise_for_status()
    # logger.debug("head %s, status-code: %s", url, resp.status_code)
    return resp


def get_modified_key(headers):
    if "ETag" in headers:
        etag = headers["ETag"]
        assert etag[0] == etag[-1] == '"'
        return etag[1:-1], True
    elif "X-Devpi-Uuid" in headers:
        uuid = headers["X-Devpi-Uuid"]
        serial = headers["X-Devpi-Serial"]
        return f"{uuid}:{serial}", False
    else:
        # no key, pitty
        # logger.debug("no key in headers: %s", headers)
        return None, False


def get_pypi_json(url, cache: DCache | None = None, refresh=False, no_cache=False):
    logger.debug(
        "get-pypi-json: %s, cache: %s, refresh: %s, no_cache: %s",
        url,
        cache,
        refresh,
        no_cache,
    )
    if no_cache:
        cached = None
    else:
        if cache is None:
            cache = DEFAULT_CACHE

        cached = cache.get(url)
        logger.debug("CACHED: %r: %s", url, bool(cached))

    if cached:
        (serial, is_etag), data = cached
        logger.debug("serial: %s, is_etag: %s", serial, is_etag)
        if refresh:
            logger.debug("refresh")
            if is_etag:
                resp: Response = fetch_json(url, etag=serial)
                if resp.status_code == 304:
                    # not modified
                    logger.debug("unchanged (etag)")
                    pass
                else:
                    serial, is_etag = get_modified_key(resp.headers)
                    data = resp.json()
                    cached = ((serial, is_etag), data)
                    cache.set(url, cached)
            else:
                resp = fetch_headers(url)
                check, _ = get_modified_key(resp.headers)
                if check != serial:
                    cached = None
        else:
            logger.debug("not refreshing")
            pass

    if not cached:
        resp = fetch_json(url)
        serial, is_etag = get_modified_key(resp.headers)
        logger.debug("serial: %s, is_etag: %s", serial, is_etag)
        data = resp.json()
        if cache is not None:
            logger.debug("CACHING: %s, serial: %s, is_etag: %s", url, serial, is_etag)
            cache.set(url, ((serial, is_etag), data))
        else:
            logger.debug("CACHE IS NONE")
            pass
    else:
        logger.debug("cached: %s", url)
        pass

    return data


def get_project_list(url, cache: DCache | None = None, refresh=False, no_cache=False):
    json = get_pypi_json(url, cache, refresh, no_cache)
    return [normalize(project["name"]) for project in json["projects"]]


def get_project_info(
    index_url, name, cache: DCache | None = None, refresh=False, no_cache=False
):
    name = normalize(name)
    index_url = index_url[:-1] if index_url[-1] == "/" else index_url
    url = f"{index_url}/{name}/"
    json = get_pypi_json(url, cache, refresh, no_cache)
    return json


def get_version_info(
    index_url, name, version, cache: DCache | None = None, refresh=False, no_cache=False
):
    name = normalize(name)
    index_url = index_url[:-1] if index_url[-1] == "/" else index_url

    if index_url == PYPI_SIMPLE_URL:
        url = f"{PYPI_API_URL}/{name}/{version}/json"
        json = get_pypi_json(url, cache, refresh, no_cache)
        return json
    else:
        wheels, sdists = get_sources_for_version(
            index_url, name, version, cache, refresh, no_cache
        )
        print("wheels:", wheels)
        print("sdists:", sdists)

        raise NotImplementedError("from wheels or sdists: %s, %s", wheels, sdists)


def get_project_version_dependencies(
    index_url, name, version, cache: DCache | None = None, refresh=False, no_cache=False
):
    json = get_version_info(index_url, name, version, cache, refresh, no_cache)
    return json["info"]["requires_dist"] or []


ALLOWED_EXTENSIONS = (".whl", ".tar.gz", ".zip")


def filter_artifact(filename):
    for ext in ALLOWED_EXTENSIONS:
        if filename.endswith(ext):
            return filename[: -len(ext)], ext
    return None, None


def get_project_versions(
    index_url,
    name,
    wheel=True,
    sdist=True,
    cache: DCache | None = None,
    refresh=False,
    no_cache=False,
) -> list[(Version, VersionSpecifiers | None)]:
    json = get_project_info(index_url, name, cache, refresh, no_cache)
    logger.debug("get-project-versions: %s %s", name, index_url)
    wheel_versions = set()
    sdist_versions = set()
    for file_meta in json["files"]:
        filename = file_meta["filename"]
        requires_python = file_meta.get("requires-python")
        if requires_python:
            requires_python = VersionSpecifiers(
                normalize_python_specifier(requires_python)
            )
        else:
            requires_python = None

        artifact, ext = filter_artifact(filename)
        if not artifact:
            logger.debug("skipping invalid artifact: %s", filename)
            continue
        if ext == ".whl" and wheel:
            wheel = parse_wheel_filename(filename)
            try:
                wheel_versions.add((Version(wheel.version), requires_python))
            except Exception as error:
                logger.debug("invalid pep 440 version: %s, %s", wheel.version, error)
        elif sdist:
            version = artifact[len(name) + 1 :]
            logger.debug("artifact: %s%s, version: %s", artifact, ext, version)
            try:
                sdist_versions.add((Version(version), requires_python))
            except Exception as error:
                logger.debug("invalid pep 440 version: %s, %s", version, error)
    versions = sorted(
        wheel_versions | sdist_versions, key=lambda vr: vr[0], reverse=True
    )
    return versions


def get_sources_for_version(
    index_url, name, version, cache: DCache | None = None, refresh=False, no_cache=False
):
    json = get_project_info(index_url, name, cache, refresh, no_cache)
    wheels = []
    sdists = []
    for file_meta in json["files"]:
        filename = file_meta["filename"]
        artifact, ext = filter_artifact(filename)
        if not artifact:
            logger.debug("skipping invalid artifact: %s", filename)
            continue
        if ext == ".whl":
            wheel = parse_wheel_filename(filename)
            if wheel.version == version:
                wheels.append(file_meta)
        else:
            # logger.debug("non wheel: %s", filename)
            artifact_version = artifact[len(name) + 1 :]
            if artifact_version == version:
                sdists.append(file_meta)
    return wheels, sdists


def choose_sdist_for_version(
    index_url,
    name,
    version,
    env: MarkerEnvironment = None,
    cache: DCache | None = None,
    refresh=False,
    no_cache=False,
):
    wheels, sdists = get_sources_for_version(
        index_url, name, version, cache, refresh, no_cache
    )
    if env is None:
        env = MarkerEnvironment.current()
    candidates = []
    for sdist in sdists:
        logger.debug("sdist candidate: %s", sdist)
        sdist["filename"]
        candidates.append(sdist)

    if candidates:
        return candidates[0]
    return None


def choose_wheel_for_version(
    index_url,
    name,
    version,
    env: MarkerEnvironment = None,
    cache: DCache | None = None,
    refresh=False,
    no_cache=False,
):
    wheels, sdists = get_sources_for_version(
        index_url, name, version, cache, refresh, no_cache
    )
    if env is None:
        env = MarkerEnvironment.current()

    for wheel in wheels:
        logger.debug("wheel candidate: %s", wheel)
        req = wheel_tags_to_requirements(name, parse_wheel_filename(wheel["filename"]))
        logger.debug("wheel as reqs: %s", req)
        if req.evaluate_markers(env, []):
            return wheel
    return None


def show_headers(resp: Response):
    for h in resp.headers:
        print(h, resp.headers[h])
