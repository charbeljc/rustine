from httpx import AsyncClient
from rustine.tools import Cache, normalize
import logging
from pubgrub import Version

USER_AGENT = "rustine/0.0.1-dev0+meow <charbeljacquin@gmail.com>"

logger = logging.getLogger(__name__)


async def get_releases_raw(
    client: AsyncClient, project: str, cache: Cache, refresh: bool = False
) -> list[str]:
    assert "/" not in normalize(project)
    url = (
        f"https://pypi.org/simple/{normalize(project)}/"
        + "?format=application/vnd.pypi.simple.v1+json"
    )

    # normalize removes all dots in the name
    cached = cache.get("pypi_simple_releases", normalize(project) + ".json")
    if cached and not refresh and not cache.refresh_versions:
        logger.debug(f"Using cached releases for {url}")
        return cached

    etag = cache.get("pypi_simple_releases", normalize(project) + ".etag")
    logger.debug(f"Querying releases from {url}")
    if etag:
        headers = {"user-agent": USER_AGENT, "If-None-Match": etag.strip()}
    else:
        headers = {"user-agent": USER_AGENT}

    response = await client.get(url, headers=headers)
    if response.status_code == 200:
        logger.debug(f"New response for {url}")
        data = response.text
        cache.set("pypi_simple_releases", normalize(project) + ".json", data)
        if etag := response.headers.get("etag"):
            cache.set("pypi_simple_releases", normalize(project) + ".etag", etag)
        return data
    elif response.status_code == 304:
        assert cached
        logger.debug(f"Not modified, using cached for {url}")
        return cached
    else:
        response.raise_for_status()
        raise RuntimeError(f"Unexpected status: {response.status_code}")


async def get_metadata(
    client: AsyncClient, project: str, version: Version, cache: Cache
) -> str:
    url = f"https://pypi.org/pypi/{normalize(project)}/{version}/json"

    cached = cache.get(
        "pypi_json_version_metadata", f"{normalize(project)}@{version}.json"
    )
    if cached:
        logger.debug(f"Using cached metadata for {url}")
        text = cached
    else:
        response = await client.get(url, headers={"user-agent": USER_AGENT})
        logger.debug(f"Querying metadata from {url}")
        response.raise_for_status()
        text = response.text
        cache.set(
            "pypi_json_version_metadata", f"{normalize(project)}@{version}.json", text
        )
    try:
        return text
    except Exception as err:
        raise RuntimeError(
            f"Failed to parse metadata for {project} {version}, "
            f"this is most likely a bug"
        ) from err
