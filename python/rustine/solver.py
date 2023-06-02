from __future__ import annotations
import typing

try:
    import sys

    import oxidized_importer

    finder = oxidized_importer.OxidizedFinder()
    sys.meta_path.insert(0, finder)
except ImportError:
    import sys

import asyncio

# from urllib3.util.url import parse_url
import logging
from dataclasses import dataclass

import click
import tomli
from httpx import AsyncClient, AsyncHTTPTransport
from progress.spinner import PixelSpinner as Spinner
from pubgrub import MarkerEnvironment, Requirement, Version, VersionSpecifier
from pubgrub import resolve as pubgrub_resolve
from pubgrub.provider import AbstractDependencyProvider
from rustine.tools import normalize
from rustine.package_index import get_project_versions, get_project_version_dependencies, PYPI_SIMPLE_URL, logger as index_logger

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
index_logger.setLevel(logging.INFO)

def asyncio_run(coro):
    if sys.version_info < (3, 11):
        return asyncio.run(coro)
    else:
        with asyncio.Runner() as runner:
            return runner.run(coro)


@dataclass(order=True, unsafe_hash=True)
class Package:
    name: str
    extras: set[str] | list[str] | None = None

    def __post_init__(self):
        if "[" in self.name:
            if self.extras:
                raise ValueError(
                    "Extra keyword should be used only when "
                    "not specified in package name."
                )
            name, extras = self.name.split("]")
            name = name.strip()
            extras = extras.strip()
            if extras[-1] != "]":
                raise ValueError("Unterminated extras in package specification.")
            extras = set(extra.strip() for extra in extras.split(","))
            self.name = name
            self.extras = extras

        if type(self.extras) is list:
            self.extras = set(self.extras)

        if self.name != ".":
            self.name = normalize(self.name)

    def __repr__(self):
        return f"<Package({self})>"

    def __str__(self):
        if self.extras:
            return f"{self.name}[{','.join(sorted(self.extras))}]"
        return self.name


class DependencyProvider(AbstractDependencyProvider[Package, str, str]):
    transport: AsyncHTTPTransport
    env: MarkerEnvironment
    user_dependencies: dict[tuple[Package, str], list[Requirement]]
    versions: dict[Package, list[str]]
    spinner: Spinner | None = None
    iterations: int = 0
    allow_pre: bool | list[str]

    def __init__(
        self,
        index_urls,
        env: MarkerEnvironment | None = None,
        allow_pre: bool | list[str] | None = None,
        spinner: Spinner = None,
    ):
        # self.package_finder = PackageFinder(index_urls=index_urls)
        if not env:
            self.env = MarkerEnvironment.current()
        else:
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

        def ensure(req: str | Requirement) -> Requirement:
            if type(req) is str:
                return Requirement(req)
            else:
                return typing.cast(Requirement, req)

        self.user_dependencies[(package, version)] = [
            ensure(req) for req in requirements
        ]

    def available_versions(self, package: Package):
        # TODO filter with requires_python & arch
        if package.name == '.':
            return ['0.0.0']
        def keep(v: Version):
            return self.allow_pre or not v.any_prerelease()

        versions = self.versions.get(package)
        if not versions:
            versions =  [str(v) for v in get_project_versions(PYPI_SIMPLE_URL, package.name) if keep(v)]
            self.versions[package] = versions
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
            # url = parse_url(version_spec)
            item = Package(name, frozenset(extras or [])), version_spec
            logger.debug("package: %s version: %s, dep: %s", package, version, item)
            yield item
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
            reqs = get_project_version_dependencies(PYPI_SIMPLE_URL, package.name, version)
        logger.debug("get-dependencies: raw: %s", reqs)

        return dict(self._filter_dependencies(package, version, reqs))

    # async def _fetch_metadata(self, package, version, cache):
    #     timeout = httpx.Timeout(10.0, connect=10.0)
    #     http_proxy = os.getenv("http_proxy")
    #     https_proxy = os.getenv("https_proxy")
    #     proxies = None
    #     if http_proxy and https_proxy:
    #         if http_proxy != https_proxy:
    #             proxies = {"http://": http_proxy, "https://": https_proxy}
    #         else:
    #             proxies = http_proxy
    #     else:
    #         proxies = http_proxy or https_proxy

    #     async with AsyncClient(
    #         proxies=proxies, http2=True, transport=self.transport, timeout=timeout
    #     ) as client:
    #         result = await get_metadata(client, package, version, cache)
    #     return result



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
    return solution


@click.command()
@click.option("--python", type=click.Path(), default=None)
@click.option("--output", "-o", type=click.Path(), default=None)
@click.option("--index-url", multiple=True)
@click.option("--debug", "-d", is_flag=True)
@click.option("--quiet", "-q", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--spinner/--no-spinner", is_flag=True)
@click.option("--requirements", "-r", type=click.File("r"), default=None)
@click.option("--pyproject", "-p", type=click.Path(file_okay=True, dir_okay=False), default=None)
def main(requirements, pyproject, python, output, index_url, debug, quiet, verbose, spinner):
    if debug or verbose:
        logging.basicConfig(level=logging.INFO)

        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
    reqs = []
    if not pyproject:
        pyproject = "pyproject.toml"

    if requirements is None:
        try:
            if verbose:
                print("try reading from %s" % pyproject, file=sys.stderr)
            with open(pyproject, "rb") as f:
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
            "no requirement source found (tried pyproject.toml, setup.cfg)",
            file=sys.stderr,
        )
        sys.exit(1)

    for line in requirements:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if line.startswith("-"):
            # print("TODO:", line, file=sys.stderr)
            continue
        reqs.append(line)
    solution = resolve(reqs, index_url, spinner)
    if output:
        file = open(output, "w")
    else:
        file = sys.stdout
    for p in sorted(
        (p for p in solution if p.name != "."),
        key=lambda p: p.name.lower(),
    ):
        print(f"{p}=={solution[p]}", file=file)
    if output:
        file.close()


def test():
    resolve(SAMPLE)


if __name__ == "__main__":
    main()
