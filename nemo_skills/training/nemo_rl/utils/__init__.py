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

from nemo_skills.training.nemo_rl.utils.skills_proxy import (
    # Discovery utilities
    VLLMServerConfig,
    # Proxy server factory
    create_skills_proxy_app,
    discover_vllm_server,
    set_vllm_server_url,
)

__all__ = [
    # Proxy server factory
    "create_skills_proxy_app",
    # Discovery utilities
    "VLLMServerConfig",
    "discover_vllm_server",
    "set_vllm_server_url",
]
