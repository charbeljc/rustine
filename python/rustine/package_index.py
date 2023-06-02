from platformdirs import AppDirs
from rustine.tools import normalize
from diskcache import Cache as DCache
import requests
from requests import Response
from logzero import logger
from pubgrub import Version
from pathlib import Path

from wheel_filename import parse_wheel_filename
USER_AGENT = "rustine/0.0.1-dev0+meow <charbeljacquin@gmail.com>"

CACHE_DIR = Path(AppDirs().user_cache_dir).joinpath("rustine")
DEFAULT_CACHE = DCache(CACHE_DIR)

PYPI_BASE = "https://pypi.org"

PYPI_SIMPLE_URL = "https://pypi.org/simple"
PYPI_API_URL = "https://pypi.org/pypi"

DEVPI_BASE = "http://localhost:3141/root/pypi"

def fetch_json(url, etag=None) -> Response:
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

    return resp.json(), resp.headers


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
    if no_cache:
        cached = None
    else:
        if cache is None:
            cache = DEFAULT_CACHE

        cached = cache.get(url)
        # logger.debug("CACHED: %r: %s", url, bool(cached))

    if cached:

        (serial, is_etag), data = cached
        # logger.debug("serial: %s, is_etag: %s", serial, is_etag)
        if refresh:
            # logger.debug("refresh")
            if is_etag:
                resp: Response = fetch_json(url, etag=serial)
                if resp.status_code == 304:
                    # not modified
                    # logger.debug("unchanged (etag)")
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
            # logger.debug("not refreshing")
            pass

    if not cached:

        resp = fetch_json(url)
        serial, is_etag = get_modified_key(resp.headers)
        # logger.debug("serial: %s, is_etag: %s", serial, is_etag)
        data = resp.json()
        if cache is not None:
            # logger.debug("CACHING: %s, serial: %s, is_etag: %s", url, serial, is_etag)
            cache.set(url, ((serial, is_etag), data))
        else:
            #logger.debug("CACHE IS NONE")
            pass
    else:
        #logger.debug("cached: %s", url)
        pass

    return data

def get_project_list(url, cache: DCache | None = None, refresh=False, no_cache=False):
    json = get_pypi_json(url, cache, refresh, no_cache)
    return [normalize(project['name']) for project in json['projects']]


def get_project_info(index_url, name, cache: DCache | None = None, refresh=False, no_cache=False):
    name = normalize(name)
    index_url = index_url[:-1] if index_url[-1] == '/' else index_url
    url = f"{index_url}/{name}/"
    json = get_pypi_json(url, cache, refresh, no_cache)
    return json

def get_version_info(index_url, name, version, cache: DCache | None = None, refresh=False, no_cache=False):
    name = normalize(name)
    index_url = index_url[:-1] if index_url[-1] == '/' else index_url

    if index_url == PYPI_SIMPLE_URL:
        url = f"{PYPI_API_URL}/{name}/{version}/json"
        json = get_pypi_json(url, cache, refresh, no_cache)
        return json
    else:
        wheels, sdists = get_sources_for_version(index_url, name, version, cache, refresh, no_cache)
        print("wheels:", wheels)
        print("sdists:", sdists)

        raise NotImplementedError("from wheels or sdists: %s, %s", wheels, sdists)


def get_project_version_dependencies(index_url, name, version, cache: DCache | None = None, refresh=False, no_cache=False):
    json = get_version_info(index_url, name, version, cache, refresh, no_cache)
    return json['info']['requires_dist'] or []


ALLOWED_EXTENSIONS = (
    ".whl",
    ".tar.gz",
    ".zip"
)

def filter_artifact(filename):
    for ext in ALLOWED_EXTENSIONS:
        if filename.endswith(ext):
            return filename[:-len(ext)], ext
    return None, None

def get_project_versions(index_url, name, wheel=True, sdist=False, cache: DCache | None = None, refresh=False, no_cache=False):
    json = get_project_info(index_url, name, cache, refresh, no_cache)
    if 'versions' in json:
        def valid(v):
            try:
                Version(v)
                return True
            except:
                return False
            

        return sorted((Version(v) for v in json['versions'] if valid(v)), reverse=True)
    else:
        wheel_versions = set()
        sdist_versions = set()
        for file_meta in json['files']:
            filename = file_meta['filename']
            file_meta['url']
            file_meta['hashes']
            file_meta.get("requires-python")
            artifact, ext = filter_artifact(filename)
            if not artifact:
                logger.debug("skipping invalid artifact: %s", filename)
                continue
            if ext == '.whl' and wheel:
                wheel = parse_wheel_filename(filename)
                try:
                    wheel_versions.add(Version(wheel.version))
                except:
                    pass
            elif sdist:
                version = artifact[len(name)+1:]
                logger.debug("artifact: %s%s, version: %s", artifact, ext, version)
                try:
                    sdist_versions.add(Version(version))
                except:
                    pass
        return sorted(wheel_versions | sdist_versions, reverse=True)

def get_sources_for_version(index_url, name, version, cache: DCache | None = None, refresh=False, no_cache=False):
    json = get_project_info(index_url, name, cache, refresh, no_cache)
    wheels = []
    sdists = []
    for file_meta in json['files']:
        filename = file_meta['filename']
        file_meta['url']
        file_meta['hashes']
        file_meta.get("requires-python")
        artifact, ext = filter_artifact(filename)
        if not artifact:
            logger.debug("skipping invalid artifact: %s", filename)
            continue
        if ext == '.whl':
            wheel = parse_wheel_filename(filename)
            if wheel.version == version:
                wheels.append(file_meta)
        else:
            artifact_version = artifact[len(name)+1:]
            if artifact_version == version:
                sdists.append(file_meta)
    return wheels, sdists

            
def show_headers(resp: Response):
    for h in resp.headers:
        print(h, resp.headers[h])