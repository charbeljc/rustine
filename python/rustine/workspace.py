import os
import sys
from pathlib import Path
from typing import Iterator

import click
import tomlkit
from tomlkit import TOMLDocument
import git

from pep508_rs import Requirement


def climb(
    directory: Path | None = None, boundary: Path | None = None
) -> Iterator[Path]:
    if not directory:
        directory = Path(os.path.curdir).absolute()
    if not directory.exists():
        raise ValueError(f"directory not found: {directory}")
    if directory.is_file():
        directory = directory.parent

    while directory != boundary and directory != directory.parent:
        yield directory
        directory = directory.parent
    yield directory


def locate_file(
    filename: str,
    directory: Path | None = None,
    boundary: Path | None = None,
    allow_links=False,
) -> Path | None:
    for path in climb(directory, boundary):
        probe = path.joinpath(filename)
        if (
            probe.is_file()
            or allow_links
            and probe.is_symlink()
            and probe.readlink().is_file()
        ):
            return probe
    return None


def locate_dir(
    filename: str,
    directory: Path | None = None,
    boundary: Path | None = None,
    allow_links=False,
    climbing=False,
) -> Path | None:
    for path in climb(directory, boundary):
        probe = path.joinpath(filename)
        if (
            probe.is_dir()
            or allow_links
            and probe.is_symlink()
            and probe.readlink().is_dir()
        ):
            return probe
        if not climbing:
            break
    return None


@click.command()
@click.argument("mode", type=click.Choice(choices=["dev", "prod"]))
def main(mode):
    print(f"applying mode: {mode}", file=sys.stderr)
    gitdir = locate_dir(".git", climbing=True)
    if not gitdir:
        print("no .git dir found", file=sys.stderr)
        sys.exit(1)
    repo = git.Repo(gitdir.parent)

    submodules = dict()
    for module in repo.submodules:
        name = module.name
        url = module.url
        path = module.path
        if name == path:
            name = os.path.basename(path)
        submodules[url] = module

    edit_workdir_metadata(repo, Path(repo.working_dir), submodules, mode)

    for module in repo.submodules:
        edit_submodule_metadata(repo, module, submodules, mode)


def edit_submodule_metadata(repo: git.Repo, module, submodules, mode):
    workspace = Path(repo.working_dir)
    workdir = workspace.joinpath(module.path)
    edit_workdir_metadata(repo, workdir, submodules, mode)


def edit_workdir_metadata(repo: git.Repo, workdir: Path, submodules, mode):
    for meta, kind in (
        (locate_file("Cargo.toml", workdir), "cargo"),
        (locate_file("pyproject.toml", workdir), "pyproject"),
    ):
        if meta:
            print(f"(not) editing {kind} {meta}, mode={mode}")
            doc = tomlkit.parse(open(meta).read())
            if kind == "cargo":
                cargo_edit(doc, repo, workdir, meta, submodules, mode)
            elif kind == "pyproject":
                pyproject_edit(doc, repo, workdir, meta, submodules, mode)


def cargo_edit(doc: TOMLDocument, repo: git.Repo, workdir: Path, meta: Path, submodules, mode):
    workspace = Path(repo.working_dir)
    deps = doc.get("dependencies")
    edited = False
    if not deps:
        ws = doc.get("workspace")
        if not ws:
            print("no deps, not a workspace, nothing todo")
            return
        members = ws["members"]
        edit_workspace_metadata(repo, workdir, members, submodules, mode)
        return
    for dep in deps:
        spec = deps[dep]
        if isinstance(spec, dict):
            git_url = spec.get("git")
            pth = spec.get("path")
            br = spec.get("branch")
            candidate = submodules.get(git_url)
            if not candidate:
                # print("not a submodule")
                continue
            print(f"dep  {mode}: {dep} {spec}")
            if mode == "prod" and not git_url and pth:
                print("TODO dev -> prod")
            elif mode == "dev" and git_url and not pth:
                newspec = tomlkit.inline_table()
                newspec.add("path", str(workspace.joinpath(candidate.path)))
                for key, value in spec.items():
                    if key in ("git", "branch"):
                        continue
                    newspec.add(key, value)
                deps[dep] = newspec
                edited = True
        else:
            # print(f"dep simple: {dep} {spec}")
            pass
    if edited:
        print(f"EDITED {meta}")
        with open(meta, "w") as out:
            tomlkit.dump(doc, out)


def pyproject_edit(doc: TOMLDocument, repo: git.Repo, workdir, meta, submodules, mode):
    edited = False
    workspace = Path(repo.working_dir)
    if "project" not in doc:
        print(f"XXX no project, skip {meta}")
        return
    deps = doc["project"].get("dependencies")
    if not deps:
        print(f"no deps, not a workspace, strange: {meta}")
        return
    new_deps = tomlkit.array()
    for dep in deps:
        req = Requirement(dep)
        print(f"{req.name}: {req.version_or_url}")
        if type(req.version_or_url) is str:
            print("BINGO!")
            url: str = req.version_or_url
            if url.startswith("git+"):
                url, _, branch = url.removeprefix("git+").partition("@")
                candidate = submodules.get(url)
                if not candidate:
                    candidate = submodules.get(url + ".git")
                print(f"U: {url} CA: {candidate}")
                if not candidate:
                    new_deps.add_line(dep)
                else:
                    path = workspace.joinpath(candidate.path)
                    new_url = f"file://{path}"
                    old_req = str(req)
                    old_req, sc, markers = old_req.partition(";")
                    package, a, url = old_req.partition("@")
                    package = package.strip()
                    url = url.strip()
                    if sc:
                        markers = f"; {markers}"
                    else:
                        markers = ""
                    new_req = f"{package} @ {new_url}{markers}"
                    new_deps.add_line(new_req)
                    edited = True
            else:
                new_deps.add_line(dep)
        else:
            new_deps.add_line(dep)
    new_deps.add_line()
    if edited:
        doc["project"]["dependencies"] = new_deps
    if edited:
        print(f"EDITED {meta}")
        with open(meta, "w") as out:
            tomlkit.dump(doc, out)


def poetry_to_req(doc: TOMLDocument, dev=False):
    deps = doc["tool"]["poetry"]["dependencies"]
    for package, version in deps.items():
        if package == "python":
            continue
        if not version:
            print("no version: ", package)
            continue
        if isinstance(version, str):
            print(package, version.replace("^", "~="))
        else:
            v = version.get("version")
            if not v:
                assert "git" in version
            else:
                extras = version.get("extras")
                if extras:
                    package = package + "[" + ",".join(extras) + "]"
                print(package, v.replace("^", "~="))


def edit_workspace_metadata(repo: git.Repo, workdir: Path, members: list[str], submodules, mode):
    for member in members:
        edit_workdir_metadata(repo, workdir.joinpath(member), submodules, mode)


if __name__ == "__main__":
    main()
