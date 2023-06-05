from __future__ import annotations

import email
import json
from pathlib import Path
import subprocess
import typing
from urllib.parse import urlparse

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
from logzero import logger
from progress.spinner import PixelSpinner as Spinner
from pubgrub import (
    MarkerEnvironment,
    Requirement,
    Version,
    VersionSpecifier,
    VersionSpecifiers,
)
from pubgrub import resolve as pubgrub_resolve
from pubgrub.provider import AbstractDependencyProvider

from rustine.package_index import (
    choose_sdist_for_version,
    choose_wheel_for_version,
    fetch_sdist_metadata,
    fetch_wheel_metadata,
    get_project_versions,
    DEFAULT_CACHE,
)
from rustine.tools import normalize


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
    index_urls: list[str]
    env: MarkerEnvironment
    user_dependencies: dict[tuple[Package, str], list[Requirement]]
    source_packages: dict[Package, str]
    versions: dict[Package, list[str]]
    spinner: Spinner | None = None
    iterations: int = 0
    allow_pre: bool | list[str]
    refresh: bool
    no_cache: bool

    def __init__(
        self,
        index_urls,
        env: MarkerEnvironment | None = None,
        allow_pre: bool | list[str] | None = None,
        refresh: bool = False,
        no_cache: bool = False,
        spinner: Spinner = None,
    ):
        # self.package_finder = PackageFinder(index_urls=index_urls)
        if not env:
            self.env = MarkerEnvironment.current()
        else:
            self.env = env

        self.user_dependencies = dict()
        self.versions = dict()
        self.source_packages = dict()

        if spinner:
            self.spinner = Spinner()
        else:
            self.spinner = None

        allow_pre = allow_pre or False
        self.allow_pre = allow_pre
        self.refresh = refresh
        self.no_cache = no_cache

        if not index_urls:
            index_urls = ["https://pypi.org/simple"]
        self.index_urls = index_urls

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

    def remove_dependencies(self, package: Package, version: str):
        logger.debug("remove-dependencies: (%r, %r)", package, version)

        del self.user_dependencies[(package, version)]

    def register_package_url(self, package: Package, url: str):
        logger.warning("register-package-url: %s, %s", package, url)
        self.source_packages[package] = url

    def available_versions(self, package: Package, refresh=None, no_cache=None):
        logger.debug("available-versions: %s", package.name)
        if refresh is None:
            refresh = self.refresh
        if no_cache is None:
            no_cache = self.no_cache

        if package.name == ".":
            return ["0.0.0"]

        if package in self.source_packages:
            return ["0.1.0"]

        python_version = self.env.python_version.version

        def keep(v: Version, requires_python):
            if v.any_prerelease() and not self.allow_pre:
                logger.debug("skipping prerelease: %s %s", package.name, v)
                return False
            if requires_python:
                try:
                    requires_python = VersionSpecifiers(requires_python)
                except Exception as error:
                    logger.warning(
                        "%s: could not parse python version specifier: %s %s",
                        package.name,
                        requires_python,
                        error,
                    )
                    # return True
                    raise
                if not any(vs.contains(python_version) for vs in requires_python):
                    logger.debug(
                        "version %s for package %s not compatible with %s (%s)",
                        v,
                        package,
                        python_version,
                        requires_python,
                    )
                    return False
            return True

        versions = self.versions.get(package)
        if versions:
            # logger.debug("got versions from cache: %s", package)
            return versions

        all_versions = get_project_versions(
            self.index_urls[0],
            package.name,
            refresh=refresh,
            no_cache=no_cache,
        )
        logger.debug("all version: %s", all_versions)
        versions = [
            str(v) for (v, requires_python) in all_versions if keep(v, requires_python)
        ]
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
            dep_package = Package(name, frozenset(extras or []))

            item = dep_package, version_spec
            logger.warning("package: %s version: %r, dep: %s", package, version, item)
            self.register_package_url(dep_package, version_spec)
            yield item
        elif type_ is list:
            version_spec = [vs2tuple(vs) for vs in version_spec or []]
            item = Package(name, frozenset(extras or [])), version_spec
            logger.debug("package: %s version: %r, dep: %s", package, version, item)
            yield item
        else:
            assert version_spec is None
            logger.debug("package: %s version: %r", package, version_spec)
            item = Package(name, frozenset(extras or [])), []
            yield item

    def get_dependencies(
        self, package: Package, version: Version, refresh=False, no_cache=False
    ):
        version = str(version)
        logger.debug("get-dependencies: %r, %r", package, version)

        if package.name == ".":
            reqs = self.user_dependencies.get((package, str(version)), [])

        elif package in self.source_packages:
            url = urlparse(self.source_packages[package])
            if url.scheme == "file":
                path = Path(url.path)
                if not path.is_dir():
                    raise FileNotFoundError(path)
                pyproject = path.joinpath("pyproject.toml")
                with open(pyproject, "rb") as f:
                    data = tomli.load(f)
                return data["project"].get("dependencies", [])
            else:
                raise NotImplementedError(
                    f"TODO: get dependency for {package}, "
                    f"url: {self.source_packages[package]}"
                )

        else:
            wheel = choose_wheel_for_version(
                self.index_urls[0],
                package.name,
                version,
                env=self.env,
                refresh=refresh,
                no_cache=no_cache,
            )
            if not wheel:
                sdist = choose_sdist_for_version(
                    self.index_urls[0],
                    package.name,
                    version,
                    env=self.env,
                    refresh=refresh,
                    no_cache=no_cache,
                )
                data = DEFAULT_CACHE.get(sdist["url"])
                if not data:
                    data = fetch_sdist_metadata(
                        package.name, version, sdist["filename"], sdist["url"]
                    )
                    DEFAULT_CACHE.set(sdist["url"], data)

            else:
                data = DEFAULT_CACHE.get(wheel["url"])
                if not data:
                    data = fetch_wheel_metadata(
                        package.name, version, wheel["filename"], wheel["url"]
                    )
                    DEFAULT_CACHE.set(wheel["url"], data)

            message = email.message_from_bytes(data)
            reqs = message.get_all("requires-dist")
            if reqs is None:
                reqs = []

            # reqs = get_project_version_dependencies(
            #   PYPI_SIMPLE_URL,
            #   package.name, version, refresh=refresh, no_cache=no_cache
            # )
        return list(self._filter_dependencies(package, version, reqs))

    def resolve(self, dependencies: list[str | Requirement]):
        p = Package(".")
        v = "0.0.0"
        self.add_dependencies(p, v, dependencies)
        if self.spinner:
            self.spinner.start()
        try:
            solution = pubgrub_resolve(self, p, v)
            for package in solution:
                if package in self.source_packages:
                    solution[package] = self.source_packages[package]
            return solution
        finally:
            self.remove_dependencies(p, v)
            if self.spinner:
                self.spinner.finish()

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
    if marker:
        marker = normalize_python_specifier(marker)

        sreq, _ = str(req).split(";")
        sreq = sreq.strip()
        req = Requirement(f"{sreq}; {marker}")
        # print(f"XXX: {req}")
    return req


def normalize_python_specifier(marker: str):
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
        versions = [Version(v) for v in versions[1:-1].split()]
        marker = connector.join(
            f"{symbol} {operator} {delim}{version}{delim}" for version in versions
        )
    return marker


# start = time.time()
# from pep440_rs import PreRelease
def transform(a, v):
    if a == "pre" and type(v) == tuple and len(v) == 2:
        k, i = v
        return str(k), i
    return v


VERSION_ATTRS = ["dev", "epoch", "post", "pre", "release"]


def vs2tuple(vs: VersionSpecifier | str):
    if type(vs) is str:
        try:
            vs = VersionSpecifier(vs)
        except ValueError:
            print(f"PEP440: {vs}")
            raise
    spec = (str(vs.operator), vs.version)
    return spec


SAMPLE = [
    "ipython",
    "pytest",
    "clap",
    "black[d,jupyter]",
]


@click.command()
@click.option("--debug", "-d", is_flag=True)
@click.option("--quiet", "-q", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--index-url", multiple=True)
@click.option("--python", type=click.Path(), default=None)
@click.option("--pre", is_flag=True)
@click.option("--pip-args")
@click.option("--requirements", "-r", type=click.File("r"), default=None)
@click.option(
    "--pyproject", "-p", type=click.Path(file_okay=True, dir_okay=False), default=None
)
@click.option("--output", "-o", type=click.Path(), default=None)
@click.option("--spin/--no-spin", is_flag=True)
def main(
    requirements,
    pyproject,
    python,
    pre,
    output,
    index_url,
    debug,
    quiet,
    pip_args,
    verbose,
    spin,
):
    logging.basicConfig(level=logging.INFO)
    logger.setLevel(logging.INFO)
    if debug:
        logger.setLevel(logging.DEBUG)
    reqs = []
    if not pyproject:
        pyproject = "pyproject.toml"
    logger.debug("pip_args: %s", pip_args)
    logger.debug("python: %s", python)
    logger.debug("output: %r", output)

    if python:
        if python.startswith('"') and python.endswith('"'):
            python = python[1:-1]
        env = get_markers_from_executable(python)
    else:
        env = MarkerEnvironment.current()

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
        reqs.append(line),

    provider = DependencyProvider(index_url, env, allow_pre=pre, spinner=spin)

    solution = provider.resolve(reqs)

    if output:
        file = open(output, "w")
    else:
        file = sys.stdout
    for p in sorted(
        (p for p in solution if p.name != "."),
        key=lambda p: p.name.lower(),
    ):
        if type(solution[p]) is str:
            print(f"{p} @ {solution[p]}", file=file)
        else:
            print(f"{p}=={solution[p]}", file=file)
        if verbose and file != sys.stdout:
            if type(solution[p]) is str:
                print(f"{p} @ {solution[p]}", file=sys.stderr)
            else:
                print(f"{p}=={solution[p]}", file=sys.stderr)

    if output:
        file.flush()
        file.close()


CAPTURE_MARKERS_SCRIPT = """
import os
import sys
import platform
import json
def format_full_version(info):
    version = '{0.major}.{0.minor}.{0.micro}'.format(info)
    kind = info.releaselevel
    if kind != 'final':
        version += kind[0] + str(info.serial)
    return version

if hasattr(sys, 'implementation'):
    implementation_version = format_full_version(sys.implementation.version)
    implementation_name = sys.implementation.name
else:
    implementation_version = '0'
    implementation_name = ''
bindings = {
    'implementation_name': implementation_name,
    'implementation_version': implementation_version,
    'os_name': os.name,
    'platform_machine': platform.machine(),
    'platform_python_implementation': platform.python_implementation(),
    'platform_release': platform.release(),
    'platform_system': platform.system(),
    'platform_version': platform.version(),
    'python_full_version': platform.python_version(),
    'python_version': '.'.join(platform.python_version_tuple()[:2]),
    'sys_platform': sys.platform,
}
json.dump(bindings, sys.stdout)
sys.stdout.flush()
"""


def get_markers_from_executable(python):
    p = subprocess.run(
        [python],
        input=CAPTURE_MARKERS_SCRIPT,
        text=True,
        capture_output=True,
        check=True,
    )
    return MarkerEnvironment(**json.loads(p.stdout))


def test():
    DependencyProvider().resolve(SAMPLE)


if __name__ == "__main__":
    main()
