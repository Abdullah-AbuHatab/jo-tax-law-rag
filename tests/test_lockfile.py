"""The Docker image installs requirements.lock (hashed, full closure); requirements.txt is the edited list."""

import re

from app.config import REPO_ROOT

LOCK = REPO_ROOT / "requirements.lock"


def _name(requirement: str) -> str:
    return re.sub(r"[-_.]+", "-", requirement.split("[", 1)[0]).lower()


def _pins(text: str) -> dict[str, str]:
    pins = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._\[\],-]*)==([^\s;\\]+)", line)
        if match:
            pins[_name(match.group(1))] = match.group(2)
    return pins


def test_lock_pins_every_top_level_requirement_at_the_same_version():
    top_level = _pins((REPO_ROOT / "requirements.txt").read_text(encoding="utf-8"))
    locked = _pins(LOCK.read_text(encoding="utf-8"))
    assert top_level, "requirements.txt has no pins"
    stale = {name: (version, locked.get(name)) for name, version in top_level.items() if locked.get(name) != version}
    assert stale == {}, f"requirements.lock is stale (edit requirements.txt, then re-run pip-compile): {stale}"
    assert len(locked) > len(top_level)  # the transitive closure is pinned too


def test_every_locked_package_carries_a_hash():
    entries = re.split(r"\n(?=[A-Za-z0-9])", LOCK.read_text(encoding="utf-8"))
    packages = [entry for entry in entries if re.match(r"[A-Za-z0-9]", entry)]
    assert packages
    assert [entry.split()[0] for entry in packages if "--hash=sha256:" not in entry] == []


def test_image_and_ci_use_the_lock():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "pip install --require-hashes --no-deps -r requirements.lock" in dockerfile
    assert "-r requirements.txt" not in dockerfile
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "pip-compile" in ci and "git diff --exit-code requirements.lock" in ci
    assert "pip-audit -r requirements.lock" in ci


def test_pgvector_image_is_pinned_by_digest_everywhere_it_runs():
    import yaml

    ci = yaml.safe_load((REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    images = [job["services"]["postgres"]["image"] for job in ci["jobs"].values() if "services" in job]
    images += [service["image"] for service in compose["services"].values() if "pgvector" in service.get("image", "")]
    assert len(images) == 2
    digests = {image.split("@", 1)[1] for image in images}
    assert len(digests) == 1 and re.fullmatch(r"sha256:[0-9a-f]{64}", digests.pop()), images
    assert all(image.startswith("pgvector/pgvector:pg16@") for image in images)


def test_dependabot_watches_the_tools_requirements_too():
    import yaml

    config = yaml.safe_load((REPO_ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8"))
    pip_dirs = [u["directory"] for u in config["updates"] if u["package-ecosystem"] == "pip"]
    # The "/" entry also scans tools/requirements.txt; a second "/tools" entry opened every
    # tools bump twice, so exactly one pip entry is expected.
    assert pip_dirs == ["/"]
    assert (REPO_ROOT / "tools" / "requirements.txt").exists()
    assert all(u["cooldown"]["default-days"] == 7 for u in config["updates"])
