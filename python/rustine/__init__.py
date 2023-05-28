from __future__ import annotations

try:
    import oxidized_importer
    import sys

    finder = oxidized_importer.OxidizedFinder()
    sys.meta_path.insert(0, finder)
except ImportError:
    pass
import io
import asyncio
from dataclasses import dataclass
from pathlib import Path
from progress.spinner import PixelSpinner as Spinner
import json_stream
import click
import httpx
from httpx import AsyncClient, AsyncHTTPTransport
import logging

from resolve_prototype.common import (
    Cache,
)
from resolve_prototype.package_index import (
    logger as monotrail_logger,
    get_metadata,
    get_releases,
    get_releases_raw,
)
from resolve_prototype.resolve import parse_requirement_fixup
from resolve_prototype.common import normalize

logger = logging.getLogger(__name__)
monotrail_logger.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
# import pep440_rs
# from pep508_rs import Requirement, VersionSpecifier, Version as BaseVersion, PreRelease, MarkerEnvironment
from pubgrub import (
    VersionSpecifier,
    resolve as pubgrub_resolve,
    Version,
    MarkerEnvironment,
    Requirement,
)

# root_requirements = [Requirement("clap"), Requirement("pylint"), Requirement("pylint")]

# requires_python = VersionSpecifier("<=3.8")


def asyncio_run(coro):
    if sys.version_info < (3, 11):
        return asyncio.run(coro)
    else:
        with asyncio.Runner() as runner:
            return runner.run(coro)


@dataclass(order=True, unsafe_hash=True)
class Package:
    name: str
    extras: set[str] | None = None

    def __post_init__(self):
        if self.name != ".":
            self.name = normalize(self.name)

    def __repr__(self):
        return f"<Package({self})>"

    def __str__(self):
        if self.extras:
            return f"{self.name}[{','.join(sorted(self.extras))}]"
        return self.name


class DependencyProvider:
    transport: AsyncHTTPTransport
    cache: Cache
    env: MarkerEnvironment
    user_dependencies: dict[tuple[Package, Version], list[Requirement]]
    versions: dict[str, list[Version]]
    spinner: Spinner | None = None
    iterations: int = 0
    allow_pre: bool | list[str]

    def __init__(
        self,
        index_urls,
        env: MarkerEnvironment = None,
        allow_pre: bool | list[str] | None = None,
        spinner: Spinner = None,
    ):
        # self.package_finder = PackageFinder(index_urls=index_urls)
        self.cache = Cache(
            Path("~/.cache/rustine").expanduser(), refresh_versions=False
        )
        if not env:
            env = MarkerEnvironment.current()
        self.env = env
        self.user_dependencies = dict()
        self.client = AsyncClient()
        self.versions = dict()
        self.spinner = spinner
        allow_pre = allow_pre or False
        self.allow_pre = allow_pre
        self.transport = AsyncHTTPTransport(retries=3)

    def should_cancel(self):
        # get a chance to catch KeyboardInterrupt
        self.iterations += 1
        try:
            if self.spinner:
                self.spinner.next()
        except KeyboardInterrupt:
            print("Interrupted!")
            raise

    def add_dependencies(
        self, package: Package, version: str, requirements: list[Requirement | str]
    ):
        logger.debug("add-dependencies: (%r, %r) : %r", package, version, requirements)
        self.user_dependencies[(package, version)] = [
            Requirement(r) if type(r) is str else r for r in requirements
        ]

    # def choose_package_version(self, potential_packages):
    #     package, version_range = potential_packages[0]
    #     if package == ".":
    #         return ".", Version("0.0.0")

    #     NotImplementedError("choose_package_version")

    def available_versions(self, package: Package):
        cached = self.versions.get(package)
        if cached:
            return cached
        versions = self._fetch_available_versions(package)
        self.versions[package] = versions
        return versions

    def _fetch_available_versions(self, package):
        if package.name == ".":
            versions = ["0.0.0"]
        else:
            retry = 2
            while retry:  # loop = asyncio.get_running_loop()
                try:
                    results = asyncio_run(
                        self._fetch_releases(package.name, self.cache)
                    )
                    break
                except RuntimeError as error:
                    print(f" ouch! {error} {package}", file=sys.stderr)
                    if retry:
                        retry -= 1
                        continue
                    print(f" die! {error} {package}", file=sys.stderr)
                    raise
            versions = []
            for v in results:
                try:
                    if Version(v).any_prerelease() and (
                        not self.allow_pre
                        or self.allow_pre is not True
                        and package.name in self.allow_pre
                    ):
                        continue
                except Exception as error:
                    # print("ouch!", package, error, file=sys.stderr)
                    continue
                versions.append(v)  # Cross binary
        versions = list(reversed(versions))
        logger.debug("available-versions: %s (%d)", package, len(versions))
        return versions

    def _filter_dependencies(
        self, package: Package, version, dependencies: list[Requirement]
    ) -> iter[tuple[Package, list]]:
        for dep in dependencies:
            yield from self._filter_dependency(package, version, dep)

    def _filter_dependency(self, package: Package, version, dep: Requirement):
        dep = Requirement(str(dep))  # Cross binary boundary
        logger.debug("filtering: (%s %s): %s", package, version, dep)
        if not dep.evaluate_markers(self.env, list(package.extras or [])):
            logger.debug("rejected by env: %s", dep)
            return
        name = dep.name
        extras = dep.extras
        version_spec = dep.version_or_url
        type_ = type(version_spec)
        if type_ is str:
            logger.warning(f"XXX: package: {package} vs: {version_spec}")
        elif type_ is list:
            version_spec = [vs2tuple(vs) for vs in version_spec or []]
            item = Package(name, frozenset(extras or [])), version_spec
            logger.debug("package: %s version: %s, dep: %s", package, version, item)
            yield item
        else:
            logger.debug("package: %s version: %s", package, version_spec)
            item = Package(name, frozenset(extras or [])), []
            yield item

    def get_dependencies(self, package: Package, version):
        version = str(version)
        logger.debug("get-dependencies: %r, %r", package, version)
        if package.name == ".":
            reqs = self.user_dependencies.get((package, str(version)), [])
        else:
            reqs = [dep for dep in self._fetch_dependencies(package, version)]
        logger.debug("get-dependencies: raw: %s", reqs)

        return dict(self._filter_dependencies(package, version, reqs))

    def _fetch_dependencies(self, package: Package, version):
        retry = 2
        # loop = asyncio.get_running_loop()
        while retry:
            try:
                result = asyncio_run(
                    self._fetch_metadata(package.name, version, self.cache)
                )
                break
            except RuntimeError as error:
                if retry:
                    retry -= 1
                    continue
                print(f" error: {error} {package} {version}")
                raise

        for req in result.requires_dist or []:
            req = parse_requirement_fixup(req, None)
            req = normalize_requirement(req)
            yield req

    async def _fetch_metadata(self, package, version, cache):
        timeout = httpx.Timeout(10.0, connect=10.0)
        async with AsyncClient(
            http2=True, transport=self.transport, timeout=timeout
        ) as client:
            result = await get_metadata(client, package, version, cache)
        return result

    async def _fetch_releases(self, package, cache):
        timeout = httpx.Timeout(10.0, connect=10.0)
        async with AsyncClient(
            http2=True, transport=self.transport, timeout=timeout
        ) as client:
            raw = await get_releases_raw(client, package, cache)
            data = json_stream.load(io.StringIO(raw))
            versions = list(data["versions"])
        return versions


def normalize_requirement(req):
    marker: str | None = req.marker
    if marker and (" in " in marker):
        connector = " or "
        operator = "=="
        symbol, versions = marker.split(" in ")
        if symbol.endswith(" not"):
            connector = " and "
            operator = "!="
        assert versions[0] in ("'", '"')
        assert versions[-1] in ("'", '"')
        delim = versions[0]
        versions = [Version(v) for v in versions[1:-1].split()]
        marker = connector.join(
            f"{symbol} {operator} {delim}{version}{delim}" for version in versions
        )
        sreq, _ = str(req).split(";")
        sreq = sreq.strip()
        req = Requirement(f"{sreq}; {marker}")
        # print(f"XXX: {req}")
    return req


# start = time.time()
# from pep440_rs import PreRelease
def transform(a, v):
    # print(f"XXX transform: {a!r}, {v!r}")
    if a == "pre" and type(v) == tuple and len(v) == 2:
        k, i = v
        return str(k), i
    return v


VERSION_ATTRS = ["dev", "epoch", "post", "pre", "release"]


def vs2tuple(vs: VersionSpecifier):
    if type(vs) is str:
        try:
            vs = VersionSpecifier(vs)
        except ValueError:
            print(f"PEP440: {vs}")
            raise
    asdict = {a: transform(a, getattr(vs.version, a)) for a in VERSION_ATTRS}

    try:
        version = Version(**asdict)
    except Exception:
        # breakpoint()
        raise
    spec = (str(vs.operator), version)
    return spec


SAMPLE = [
    "ipython",
    "pytest",
    "clap",
    "black[d,jupyter]",
    # "transformers[torch,sentencepiece,tokenizers,torch-speech,vision,integrations,timm,torch-vision,codecarbon,accelerate,video]",
]


def resolve(
    dependencies: list[str], index_urls: list[str] = None, spinner: bool = None
):
    if spinner:
        spinner = Spinner()
    else:
        spinner = None
    if not index_urls:
        index_urls = ["https://pypi.org/simple"]

    dp = DependencyProvider(index_urls, spinner=spinner)
    p = Package(".")
    v = "0.0.0"
    dp.add_dependencies(p, v, dependencies)
    if spinner:
        spinner.start()
    solution = pubgrub_resolve(dp, p, v)
    if spinner:
        spinner.finish()
    for p in sorted(
        (p for p in solution if p.name != "."),
        key=lambda p: p.name.lower(),
    ):
        print(f"{p}=={solution[p]}")
    return solution


import tomli


@click.command()
@click.option("--requirements", "-r", type=click.File("r"), default=None)
@click.option("--index-url", multiple=True)
@click.option("--debug", "-d", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--spinner/--no-spinner", is_flag=True)
def main(requirements, index_url, debug, verbose, spinner):
    print(f"XXX: {requirements!r}", file=sys.stderr)
    if debug or verbose:
        logging.basicConfig(level=logging.INFO)

        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
    reqs = []
    if requirements is None:
        try:
            if verbose:
                print("try reading from pyproject.toml", file=sys.stderr)
            with open("pyproject.toml", "rb") as f:
                project = tomli.load(f)
            requirements = project["project"].get("dependencies", [])
        except Exception as error:
            print(f"error: {error}", file=sys.stderr)
    if requirements is None:
        try:
            if verbose:
                print("try reading from setup.cfg", file=sys.stderr)
            with open("setup.cfg", "rb") as f:
                project = tomli.load(f)
            requirements = project["project"]["dependencies"]
        except Exception as error:
            print(f"error: {error}", file=sys.stderr)
    if requirements is None:
        print(
            "no requirement source found (tried pyproject.toml, setup.cfg)", file=sys.stderr
        )
        sys.exit(1)

    for line in requirements:
        line = line.strip()
        if line.startswith("#"):
            continue
        if line.startswith("-"):
            print("TODO:", line, file=sys.stderr)
            continue
        reqs.append(line)
    resolve(reqs, index_url, spinner)


def test():
    resolve(SAMPLE)


if __name__ == "__main__":
    main()
