# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import os
import shlex
import subprocess
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import yaml

from nemo_skills.pipeline.utils import (
    cluster_download_file,
    cluster_path_exists,
    cluster_upload,
    create_remote_directory,
    get_cluster_config,
    get_tunnel,
)
from nemo_skills.pipeline.utils.mounts import get_mounts_from_config

_SUPPORTED_CLUSTER_CONFIG_MODES = {"assert", "overwrite", "reuse"}
_DEFAULT_CLUSTER_CONFIG_FILENAME = "cluster_config.yaml"
_DEFAULT_COMMIT_FILENAME = "nemo_skills_commit.json"
_UNCOMMITTED_ENV = "NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK"
_UNCOMMITTED_SKIP_VALUES = {"1", "true", "yes"}
_UNCOMMITTED_ERROR_MSG = (
    "The NeMo-Skills checkout you're using to launch this Slurm test has uncommitted changes.\n"
    "We snapshot the repo state into each test workspace for reproducibility, but we cannot do so "
    "while the working tree is dirty.\n"
    "Please commit or stash your changes, or set NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK=1 "
    "if you intentionally want to snapshot an in-progress state (note: this also disables the "
    "global nemo-skills submission check)."
)


def _is_uncommitted_check_disabled() -> bool:
    return os.environ.get(_UNCOMMITTED_ENV, "0").lower() in _UNCOMMITTED_SKIP_VALUES


@lru_cache(maxsize=1)
def _get_repo_root() -> Path:
    """Return the git repository root for the current checkout."""
    current_dir = Path(__file__).resolve().parent
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=current_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        root = result.stdout.strip()
        if root:
            return Path(root)
    # Fallback to the NeMo-Skills directory relative to this file
    return Path(__file__).resolve().parents[3]


def load_json(path):
    """Load a JSON file from the given path."""
    with open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def get_nested_value(nested_dict, nested_keys):
    for k in nested_keys:
        if not isinstance(nested_dict, dict) or k not in nested_dict:
            return None
        nested_dict = nested_dict[k]
    # resolves to the value eventually
    return nested_dict


_soft_assert_failures = []


def soft_assert(condition: bool, message: str):
    """Record a failure instead of raising immediately.

    Use this in place of `assert` when you want to collect all failures
    and report them at the end of the script.
    """
    if not condition:
        _soft_assert_failures.append(str(message))


def assert_all():
    """If any soft assertions failed, print them and exit with non-zero status.

    Does nothing if there are no failures. Intended to be called once at the end
    of a check script, before printing a success message.
    """
    if not _soft_assert_failures:
        print("ALL TESTS PASSED")
        return
    print(f"\nTEST FAILURES ({len(_soft_assert_failures)})\n")
    for i, msg in enumerate(_soft_assert_failures, 1):
        print(f"{i:3d}. {msg}")
    raise SystemExit(1)


def prepare_cluster_config_for_test(
    cluster,
    workspace,
    config_dir=None,
    *,
    cluster_config_mode: str = "assert",
    cluster_config_filename: str = _DEFAULT_CLUSTER_CONFIG_FILENAME,
    commit_metadata_filename: str = _DEFAULT_COMMIT_FILENAME,
):
    """Prepare a cluster config for testing by overriding job_dir to be within the test workspace.

    This ensures that nemo-run experiment artifacts are stored in {workspace}/nemo-run-experiments
    instead of the global cluster job_dir, making it easier to correlate experiment IDs with
    test runs and workspace locations.

    Note: This function resolves the workspace mount path to its actual filesystem path since
    job_dir needs to be set before any containers are launched (it's sent over SSH, not inside
    a container).

    Args:
        cluster: Cluster name or config dict
        workspace: Test workspace directory path (may be a mount path like /workspace/...)
        config_dir: Optional directory to search for cluster configs
        cluster_config_mode: How to handle existing snapshots inside job_dir.
            - "assert": require the saved config (and commit metadata) to match the newly generated one.
            - "overwrite": replace the saved files with the newly generated versions.
            - "reuse": load the previously saved config and use it for this run without modifying the files.
        cluster_config_filename: Name of the snapshot file saved under job_dir.
        commit_metadata_filename: Name of the commit metadata file saved under job_dir.

    Returns:
        dict: Modified cluster config with job_dir set to {workspace_source}/nemo-run-experiments
    """
    cluster_config_mode = cluster_config_mode.lower()
    if cluster_config_mode not in _SUPPORTED_CLUSTER_CONFIG_MODES:
        raise ValueError(
            f"Unsupported cluster_config_mode '{cluster_config_mode}'. "
            f"Supported values: {sorted(_SUPPORTED_CLUSTER_CONFIG_MODES)}"
        )

    # Load the cluster config
    cluster_config = get_cluster_config(cluster, config_dir)

    # Deep copy to avoid modifying original
    cluster_config = copy.deepcopy(cluster_config)

    # Resolve workspace mount path to actual source path
    # workspace might be a mount destination like /workspace/..., but job_dir needs
    # the actual filesystem path (mount source) since it's set before containers are created
    workspace_source = workspace
    if "mounts" in cluster_config:
        mounts = get_mounts_from_config(cluster_config)
        for mount in mounts:
            if ":" in mount:
                source, dest = mount.split(":", 1)
                # Check if workspace is under this mount destination
                if workspace.startswith(dest):
                    # Replace the mount destination prefix with source prefix
                    workspace_source = workspace.replace(dest, source, 1)
                    break

    # Override job_dir to be within workspace (using the resolved source path)
    test_job_dir = f"{workspace_source}/nemo-run-experiments"

    snapshot_dir = workspace_source

    if "ssh_tunnel" in cluster_config:
        cluster_config["ssh_tunnel"]["job_dir"] = test_job_dir
        job_dir = cluster_config["ssh_tunnel"]["job_dir"]
    else:
        cluster_config["job_dir"] = test_job_dir
        job_dir = cluster_config["job_dir"]

    cluster_config["job_dir"] = cluster_config.get("job_dir", test_job_dir)
    _resolve_container_image_paths(cluster_config)

    return _sync_cluster_config_snapshot(
        cluster_config,
        job_dir=job_dir,
        snapshot_dir=snapshot_dir,
        mode=cluster_config_mode,
        cluster_config_filename=cluster_config_filename,
        commit_metadata_filename=commit_metadata_filename,
    )


def _resolve_container_image_paths(cluster_config: dict):
    """Resolve local/remote symlinks for container image paths so snapshots capture canonical targets."""
    containers = cluster_config.get("containers")
    if not isinstance(containers, dict):
        return

    resolved = {}
    for name, path in containers.items():
        resolved[name] = _resolve_path_with_remote(cluster_config, path)
    cluster_config["containers"] = resolved


def _resolve_path_with_remote(cluster_config: dict, path: str):
    """Resolve the provided path locally, and fallback to remote resolution if needed."""
    if not isinstance(path, str) or not path:
        return path

    local_resolved = os.path.realpath(path)
    if os.path.exists(local_resolved):
        return local_resolved

    if cluster_config.get("executor") != "slurm":
        return local_resolved

    tunnel = None
    try:
        tunnel = get_tunnel(cluster_config)
        result = tunnel.run(f"readlink -f {shlex.quote(path)}", hide=True, warn=True)
        resolved_remote = result.stdout.strip() if result.exited == 0 else ""
        return resolved_remote or local_resolved
    except Exception:
        return local_resolved
    finally:
        if tunnel is not None:
            tunnel.cleanup()


def _sync_cluster_config_snapshot(
    cluster_config: dict,
    *,
    job_dir: str,
    snapshot_dir: str,
    mode: str,
    cluster_config_filename: str,
    commit_metadata_filename: str,
):
    """Persist the cluster config / commit metadata according to the selected mode."""
    job_dir = str(Path(job_dir))
    snapshot_dir = str(Path(snapshot_dir))
    config_remote_path = str(Path(snapshot_dir) / cluster_config_filename)
    commit_remote_path = str(Path(snapshot_dir) / commit_metadata_filename)

    if mode == "reuse":
        if not cluster_path_exists(cluster_config, config_remote_path):
            raise FileNotFoundError(
                f"cluster_config_mode 'reuse' requires an existing snapshot at {config_remote_path}"
            )
        persisted = _download_remote_yaml(cluster_config, config_remote_path)
        if not isinstance(persisted, dict):
            raise ValueError(f"Existing cluster config at {config_remote_path} is not a valid mapping.")
        _ensure_job_dir(persisted, job_dir)
        _resolve_container_image_paths(persisted)
        _sync_commit_metadata(cluster_config, commit_remote_path, mode)
        return persisted

    create_remote_directory([job_dir, snapshot_dir], cluster_config)
    existing_remote = cluster_path_exists(cluster_config, config_remote_path)
    if existing_remote:
        persisted = _download_remote_yaml(cluster_config, config_remote_path)
        if mode == "assert":
            if not _cluster_configs_equal(persisted, cluster_config):
                raise AssertionError(
                    "Existing cluster config snapshot does not match the newly generated config. "
                    "Use --cluster_config_mode overwrite to update the snapshot or reuse to keep the existing one."
                )
            _sync_commit_metadata(cluster_config, commit_remote_path, mode)
            return cluster_config

    _upload_yaml(cluster_config, cluster_config, config_remote_path)
    _sync_commit_metadata(cluster_config, commit_remote_path, mode)
    return cluster_config


def add_common_args(parser, *, include_wandb: bool = True, wandb_default: str = "nemo-skills-slurm-ci"):
    """Register the shared CLI arguments used by slurm test entrypoints."""

    parser.add_argument(
        "--workspace",
        required=True,
        help="Workspace directory containing all experiment data",
    )
    parser.add_argument(
        "--cluster",
        required=True,
        help="Cluster config name or path (same semantics as --cluster on nemo-skills CLI).",
    )
    parser.add_argument(
        "--expname_prefix",
        required=True,
        help="Experiment name prefix used to group nemo-run jobs for this test.",
    )
    if include_wandb:
        parser.add_argument(
            "--wandb_project",
            default=wandb_default,
            help="W&B project name used for logging (set to empty string to disable).",
        )

    parser.add_argument(
        "--cluster_config_mode",
        choices=sorted(_SUPPORTED_CLUSTER_CONFIG_MODES),
        default="assert",
        help="Controls how existing cluster config snapshots under the workspace job_dir are handled.",
    )

    return parser


def _cluster_configs_equal(config_a: dict, config_b: dict) -> bool:
    """Compare two configs after normalizing container image paths."""

    def _normalize(config: dict):
        config_copy = copy.deepcopy(config)
        _resolve_container_image_paths(config_copy)
        return config_copy

    return _normalize(config_a) == _normalize(config_b)


def _ensure_job_dir(cluster_config: dict, job_dir: str):
    """Ensure the provided cluster config uses the expected workspace job_dir."""
    if "ssh_tunnel" in cluster_config:
        cluster_config["ssh_tunnel"]["job_dir"] = job_dir
    else:
        cluster_config["job_dir"] = job_dir


def _sync_commit_metadata(cluster_config: dict, remote_path: str, mode: str):
    """Persist commit metadata using the same mode semantics as the config snapshot."""
    metadata = _collect_repo_metadata()
    remote_exists = cluster_path_exists(cluster_config, remote_path)

    if mode == "reuse":
        if not remote_exists:
            raise FileNotFoundError(f"cluster_config_mode 'reuse' requires existing commit metadata at {remote_path}")
        return

    if remote_exists and mode == "assert":
        if _is_uncommitted_check_disabled():
            return
        existing = _download_remote_json(cluster_config, remote_path)
        if existing != metadata:
            raise AssertionError(
                "Existing commit metadata does not match the current repository state. "
                "Use --cluster_config_mode overwrite to refresh the snapshot."
            )
        return

    _upload_json(cluster_config, metadata, remote_path)


def _collect_repo_metadata() -> dict:
    """Gather information about the current NeMo-Skills checkout."""
    repo_root = _get_repo_root()
    metadata = {
        "repo_root": str(repo_root),
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    def _run_git(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status_output = _run_git("status", "--short")
    if status_output and not _is_uncommitted_check_disabled():
        raise RuntimeError(_UNCOMMITTED_ERROR_MSG)

    metadata["commit"] = _run_git("rev-parse", "HEAD")
    metadata["describe"] = _run_git("describe", "--always", "--dirty")
    metadata["is_dirty"] = bool(status_output) if status_output is not None else None
    return metadata


def _download_remote_yaml(cluster_config: dict, remote_path: str) -> dict:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cluster_download_file(cluster_config, remote_path, tmp_path)
        with open(tmp_path, "rt", encoding="utf-8") as fin:
            return yaml.safe_load(fin) or {}
    finally:
        os.remove(tmp_path)


def _download_remote_json(cluster_config: dict, remote_path: str) -> dict:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cluster_download_file(cluster_config, remote_path, tmp_path)
        return load_json(tmp_path)
    finally:
        os.remove(tmp_path)


def _upload_yaml(cluster_config: dict, data: dict, remote_path: str):
    with tempfile.NamedTemporaryFile(mode="wt", encoding="utf-8", delete=False) as tmp:
        yaml.safe_dump(data, tmp, sort_keys=True)
        tmp_path = tmp.name
    try:
        cluster_upload(cluster_config, tmp_path, remote_path)
    finally:
        os.remove(tmp_path)


def _upload_json(cluster_config: dict, data: dict, remote_path: str):
    with tempfile.NamedTemporaryFile(mode="wt", encoding="utf-8", delete=False) as tmp:
        json.dump(data, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = tmp.name
    try:
        cluster_upload(cluster_config, tmp_path, remote_path)
    finally:
        os.remove(tmp_path)
