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

from nemo_skills.pipeline.utils.cluster import get_cluster_config
from nemo_skills.pipeline.utils.mounts import get_mounts_from_config


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


def prepare_cluster_config_for_test(cluster, workspace, config_dir=None):
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

    Returns:
        dict: Modified cluster config with job_dir set to {workspace_source}/nemo-run-experiments
    """
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

    if "ssh_tunnel" in cluster_config:
        cluster_config["ssh_tunnel"]["job_dir"] = test_job_dir
    else:
        cluster_config["job_dir"] = test_job_dir

    return cluster_config
