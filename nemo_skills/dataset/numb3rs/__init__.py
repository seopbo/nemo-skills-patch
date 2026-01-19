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

"""Numb3rs: Numbers Speech Benchmark for TN/ITN evaluation.

A speech dataset for text normalization (TN) and inverse text normalization (ITN) tasks,
containing paired written/spoken forms with corresponding synthetic audio.

Dataset: https://huggingface.co/datasets/NNstuff/Numb3rs

Categories: ADDRESS, CARDINAL, DATE, DECIMAL, DIGIT, FRACTION, MEASURE, MONEY,
           ORDINAL, PLAIN, TELEPHONE, TIME

Features:
- Dual reference evaluation: text_tn (written form) and text_itn (spoken form)
- Multiple prompt variants: neutral, explicit TN, explicit ITN
- ~10K samples, ~4.89h total duration

Usage:
    # Prepare dataset
    ns prepare_data numb3rs

    # Generate with neutral prompt, evaluate against both references
    ns generate ... \
        ++prompt_field=prompt_neutral \
        ++eval_config.reference_fields='[text_tn,text_itn]'

    # Generate with TN prompt
    ns generate ... \
        ++prompt_field=prompt_tn \
        ++eval_config.reference_fields='[text_tn,text_itn]'
"""

# Dataset configuration
DATASET_GROUP = "speechlm"
METRICS_TYPE = "audio"
DEFAULT_SPLIT = "test"

# Evaluation settings
EVAL_SPLIT = "test"
EVAL_ARGS = (
    "++eval_type=audio "
    "++eval_config.reference_fields='[text_tn,text_itn]' "  # Evaluate against both references
    # "++eval_config.normalization_mode=standard "  # Uncomment after audio-input-output-integration is merged
)

# Generation settings - OpenAI format for audio-language models
GENERATION_ARGS = (
    "++prompt_format=openai "
    "++prompt_field=prompt_neutral "  # Use neutral prompt by default
)
