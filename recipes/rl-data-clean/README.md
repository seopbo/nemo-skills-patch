# RL Data Cleaning Pipeline for IMO Proof Problems

This pipeline is designed to extract and clean high-quality IMO-level proof problems from AOPS forum data for RL training.

## Project Structure

```
rl-data-clean/
├── prompts/                    # LLM prompts for each stage
│   ├── extract-problems.yaml
│   ├── classify-if-proof.yaml
│   ├── assess-problem-quality.yaml
│   ├── assess-discussion-quality.yaml
│   ├── assess-proof-quality.yaml
│   └── assess-imo-readiness.yaml
├── scripts/                    # Postprocessing scripts
│   ├── postprocess_problem_extraction.py
│   ├── postprocess_classification.py
│   ├── postprocess_quality_assessment.py
│   ├── merge_model_results.py
│   └── compare_model_outputs.py
├── configs/                    # Pipeline configurations
│   ├── imo-proof-120b.yaml            # Single model (120B)
│   ├── imo-proof-dual-model.yaml      # Dual model validation
│   └── experiment-compare-models.yaml # Phase 1 experiment
└── pipeline/                   # Pipeline implementation
    └── proof_pipeline.py       # Enhanced pipeline code

```

## Pipeline Stages

### Stage 1: Extract Problems
- Extract math problems from raw forum discussions
- Clean and refine problem statements

### Stage 2: Classify if Proof
- Binary classification: proof vs non-proof

### Stage 3: Assess Problem Quality (NEW)
- Clarity (1-5)
- Completeness (1-5)
- Mathematical rigor (1-5)
- Difficulty estimation (P1-P6)

### Stage 4: Assess Discussion Quality (NEW)
- Has meaningful discussion (yes/no)
- Solution present (yes/no)
- Discussion coherence (1-5)

### Stage 5: Assess Proof Quality (NEW)
- Correctness (correct/incorrect/uncertain)
- Rigor (1-5)
- Elegance (1-5)
- Multiple approaches (yes/no)
- Overall quality (outstanding/excellent/good/fair/poor)

### Stage 6: Assess IMO Readiness (NEW)
- Olympiad style (yes/no/borderline)
- Pedagogical value (high/medium/low)
- Difficulty appropriateness (too_easy/appropriate/too_hard)
- IMO readiness score (0-100)

### Stage 7: Decontaminate
- Check against test sets

## Usage

### Phase 1: Model Comparison Experiment (1000 samples)
```bash
python recipes/rl-data-clean/pipeline/imo_proof_pipeline.py \
  --config experiment-compare-models
```

### Phase 2: Full Pipeline
```bash
# Single model
python recipes/rl-data-clean/pipeline/imo_proof_pipeline.py \
  --config imo-proof-120b

# Run specific stages
python recipes/rl-data-clean/pipeline/imo_proof_pipeline.py \
  --config imo-proof-120b \
  --stages classify_if_proof,assess_proof_quality
```

## Quality Thresholds

**Conservative** (300-400 problems @ 99% quality):
- Problem quality: clarity ≥ 5, completeness ≥ 5, rigor ≥ 5
- Proof quality: rigor ≥ 5, correctness = correct
- IMO readiness: score ≥ 90

**Moderate** (600-800 problems @ 95% quality):
- Problem quality: clarity ≥ 4, completeness ≥ 4, rigor ≥ 4
- Proof quality: rigor ≥ 4, correctness = correct
- IMO readiness: score ≥ 80

## Model Options

- **gpt-oss-120B**: Internal model, baseline
- **DeepSeek-V3**: Math-specialized, strong in number theory & algebra
- **Claude 3.5 Sonnet**: Excellent logical reasoning (API)
- **GPT-4o**: Strong overall math (API)
