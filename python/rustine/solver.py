from __future__ import annotations

import email
import json
from pathlib import Path
import subprocess
import typing
from typing import Optional, Union
from urllib.parse import urlparse

try:
    import sys

    import oxidized_importer

    finder = oxidized_importer.OxidizedFinder()
    sys.meta_path.insert(0, finder)
except ImportError:
    import sys


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
from rustine.tools import fixup_requirement, normalize, version_matches


@dataclass(init=False, order=True, unsafe_hash=True)
class Package:
    name: str
    extras: Optional[frozenset[str]] = None

    def __init__(self, spec: str, extras=None):
        if "[" in spec and extras is not None:
            raise TypeError("extras in name and as keyword argument")
        if "[" in spec:
            name, extras = spec.split("[")
            name = name.strip()
            extras = extras.strip()
            if extras[-1] != "]":
                raise ValueError("Unterminated extras in package specification.")
            extras = extras[:-1]
            extras = frozenset(extra.strip() for extra in extras.split(","))
        else:
            name = spec
            extras = frozenset(extras) if extras else None

        self.name = normalize(name) if name != "." else "."
        self.extras = extras

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
    source_packages: dict[Package, str]  # url
    versions: dict[Package, list[str]]  # version
    exclusions: dict[Package, VersionSpecifiers]
    spinner: Optional[Spinner] = None
    iterations: int = 0
    allow_pre: Union[bool, list[str]]
    refresh: bool
    no_cache: bool
    debug: bool
    verbose: bool

    def __init__(
        self,
        index_urls,
        env: Optional[MarkerEnvironment] = None,
        allow_pre: Optional[Union[bool, list[str]]] = None,
        refresh: bool = False,
        no_cache: bool = False,
        debug: bool = False,
        verbose: bool = False,
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

        self.debug = debug
        self.verbose = verbose

        if not index_urls:
            index_urls = ["https://pypi.org/simple"]
        self.index_urls = index_urls
        self.exclusions = dict()

    def add_dependencies(
        self, package: Package, version: str, requirements: list[Union[str, Requirement]]
    ):
        logger.debug("add-dependencies: (%r, %r) : %r", package, version, requirements)

        def ensure(req: Union[str, Requirement]) -> Requirement:
            if type(req) is str:
                return Requirement(req)
            else:
                return typing.cast(Requirement, req)

        self.user_dependencies[(package, version)] = [
            ensure(req) for req in requirements
        ]

    def exclude(self, package: Package, version: Union[str, VersionSpecifiers]):
        if isinstance(version, str):
            v = VersionSpecifiers(version)
        else:
            v = version
        self.exclusions[package] = v.to_pubgrub()

    def excluded(self, package: Package, version: Union[str,Version]):
        if package.name == "sigopt":
            # breakpoint()
            pass
        excluded_range = self.exclusions.get(package)
        if not excluded_range:
            return False
        if isinstance(version, str):
            v = Version(version)
        else:
            v = version
        excluded = excluded_range.contains(v)
        return excluded

    def remove_dependencies(self, package: Package, version: str):
        logger.debug("remove-dependencies: (%r, %r)", package, version)

        del self.user_dependencies[(package, version)]

    def resolve(self, dependencies: list[Union[str,Requirement]]):
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

    def should_cancel(self):
        # get a chance to catch KeyboardInterrupt
        self.iterations += 1
        try:
            if self.spinner:
                self.spinner.next()
        except KeyboardInterrupt:
            print("Interrupted!")
            raise

    def register_package_url(self, package: Package, url: str):
        logger.debug("register-package-url: %s, %s", package, url)
        self.source_packages[package] = url

    def available_versions(self, package: Package, refresh=None, no_cache=None):
        if self.debug and self.verbose:
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

        def keep(v: Version, requires_python: Optional[VersionSpecifiers]):
            if self.excluded(package, v):
                ## logger.info("forbidden package version: %s %s", package, v)
                return False
            if not self.allow_pre and v.any_prerelease():
                if self.debug and self.verbose:
                    logger.debug(
                        "version %s for package %s is a prerelease, skipping.",
                        v,
                        package,
                    )
                return False
            if not version_matches(python_version, requires_python):
                if self.debug and self.verbose:
                    logger.debug(
                        "version %s for package %s not compatible with %s (%s)",
                        v,
                        package,
                        python_version,
                        requires_python,
                    )
                return False
            return True

        all_versions = self.versions.get(package)
        if all_versions:
            logger.debug("got versions from cache: %s", package)
        else:
            all_versions = get_project_versions(
                self.index_urls[0],
                package.name,
                refresh=refresh,
                no_cache=no_cache,
            )
            self.versions[package] = all_versions
        versions = [
            str(v) for (v, requires_python) in all_versions if keep(v, requires_python)
        ]
        logger.debug(
            "filtered versions: %s, count, %s, last: %s, first: %s",
            package.name,
            len(versions),
            versions[0],
            versions[-1],
        )
        return versions

    def get_dependencies(
        self, package: Package, version: Version, refresh=False, no_cache=False
    ):
        version = str(version)
        logger.debug("get-dependencies: %r, %r", package, version)

        if package.name == ".":
            reqs = [
                fixup_requirement(req)
                for req in self.user_dependencies.get((package, str(version)), [])
            ]

        elif package in self.source_packages:
            url = urlparse(self.source_packages[package])
            if url.scheme == "file":
                path = Path(url.path)
                if not path.is_dir():
                    raise FileNotFoundError(path)
                pyproject = path.joinpath("pyproject.toml")
                with open(pyproject, "rb") as f:
                    data = tomli.load(f)
                return [
                    fixup_requirement(req)
                    for req in data["project"].get("dependencies", [])
                ]
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
            reqs = [
                fixup_requirement(req) for req in message.get_all("requires-dist") or []
            ]
            if reqs is None:
                reqs = []

            # reqs = get_project_version_dependencies(
            #   PYPI_SIMPLE_URL,
            #   package.name, version, refresh=refresh, no_cache=no_cache
            # )
        return list(self._filter_dependencies(package, version, reqs))

    def _filter_dependencies(
        self, package: Package, version, dependencies: list[Requirement]
    ) -> iter[tuple[Package, list]]:
        for dep in dependencies:
            yield from self._filter_dependency(package, version, dep)

    def _filter_dependency(self, package: Package, version, dep: Union[str, Requirement]):
        dep = fixup_requirement(dep)
        if self.debug and self.verbose:
            logger.debug("filtering: %s", dep)
        if not dep.evaluate_markers(self.env, list(package.extras or [])):
            if self.debug and self.verbose:
                logger.debug("dependency rejected by env: %s", dep)
            return
        name = dep.name
        extras = dep.extras
        version_spec = dep.version_or_url
        type_ = type(version_spec)
        if type_ is str:
            # url = parse_url(version_spec)
            dep_package = Package(name, frozenset(extras or []))

            item = dep_package, version_spec
            logger.debug("package: %s version: %r via: %r", name, version_spec, dep)
            self.register_package_url(dep_package, version_spec)
            yield item
        elif type_ is list:
            # breakpoint()
            # version_spec = [vs2tuple(vs) for vs in version_spec or []]
            item = Package(name, frozenset(extras or [])), version_spec
            logger.debug("package: %s version: %r via: %r", name, version_spec, dep)
            yield item
        else:
            assert version_spec is None
            logger.debug("package: %s version: %r via: %r", name, version_spec, dep)
            item = Package(name, frozenset(extras or [])), []
            yield item

def transform(a, v):
    if a == "pre" and type(v) == tuple and len(v) == 2:
        k, i = v
        return str(k), i
    return v


VERSION_ATTRS = ["dev", "epoch", "post", "pre", "release"]


def vs2tuple(vs: Union[str, VersionSpecifier]):
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
@click.option("--python", type=click.Path(path_type=str), default=None)
@click.option("--pre", is_flag=True)
@click.option("--pip-args")
@click.option("--requirements", "-r", type=click.File("r"), default=None)
@click.option(
    "--metadata", "-m", type=click.Path(file_okay=True, dir_okay=False), default=None
)
@click.option("--output", "-o", type=click.Path(), default=None)
@click.option("--spin/--no-spin", is_flag=True)
def main(
    requirements,
    metadata,
    python: str,
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
    if not metadata:
        metadata = "pyproject.toml"
    logger.debug("pip_args: %s", pip_args)
    logger.debug("python: %s", python)
    logger.debug("output: %r", output)
    if verbose:
        print("Using python:", python, file=sys.stderr)

    if python:
        if python.startswith('"') and python.endswith('"'):
            python = python[1:-1]
        env = get_markers_from_executable(python)
    else:
        env = MarkerEnvironment.current()

    if requirements is None:
        try:
            if verbose:
                print("Reading from %s" % metadata, file=sys.stderr)
            with open(metadata, "rb") as f:
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
    provider.exclude(Package("sigopt"), VersionSpecifiers("<=8.6.3"))
    solution = provider.resolve(reqs)

    if output:
        logger.debug("opening %r", output)
        file = open(output, "w")
    else:
        file = sys.stdout
    print("-e file:.", file=file)
    if verbose and file != sys.stdout:
        print("-e file:.", file=sys.stderr)
    for p in sorted(
        (p for p in solution if p.name != "."),
        key=lambda p: p.name.lower(),
    ):
        name = p.name

        if type(solution[p]) is str:
            print(f"{name} @ {solution[p]}", file=file)
        else:
            print(f"{name}=={solution[p]}", file=file)
        if verbose and file != sys.stdout:
            if type(solution[p]) is str:
                print(f"{name} @ {solution[p]}", file=sys.stderr)
            else:
                print(f"{name}=={solution[p]}", file=sys.stderr)

    file.flush()
    if output:
       file.close()
    #breakpoint()
    sys.exit(0)


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


def get_markers_from_executable(python) -> MarkerEnvironment:
    process = subprocess.run(
        [python],
        input=CAPTURE_MARKERS_SCRIPT,
        text=True,
        capture_output=True,
        check=True,
    )
    return MarkerEnvironment(**json.loads(process.stdout))


def test():
    DependencyProvider().resolve(SAMPLE)


if __name__ == "__main__":
    main()
