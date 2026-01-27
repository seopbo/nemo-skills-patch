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
- **Approach**: Detailed analysis → Binary decision (ACCEPT/REJECT)
- **Analysis**: Clarity, completeness, mathematical rigor, difficulty level
- **Output**: Detailed reasoning + ACCEPT/REJECT decision

### Stage 4: Assess Discussion Quality (NEW)
- **Approach**: Detailed analysis → Binary decision (ACCEPT/REJECT)
- **Analysis**: Meaningful content, solution presence/clarity, coherence
- **Output**: Detailed reasoning + ACCEPT/REJECT decision

### Stage 5: Assess Proof Quality (NEW)
- **Approach**: Detailed analysis → Binary decision (ACCEPT/REJECT)
- **Analysis**: Correctness, rigor & completeness, clarity, mathematical insight
- **Output**: Detailed reasoning + ACCEPT/REJECT decision

### Stage 6: Assess IMO Readiness (NEW - Final Gate)
- **Approach**: Synthesize all assessments → Binary decision (ACCEPT/REJECT)
- **Analysis**: Olympiad style, pedagogical value, difficulty, teachability, RL suitability
- **Output**: Final decision on IMO training readiness

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

## Design Philosophy: Binary Decisions

**Why Binary Instead of Numeric Scores?**

1. **More Stable**: Different models agree more easily on ACCEPT/REJECT than on "3 vs 4"
2. **Less Subjective**: No need to tune thresholds ("≥4 or ≥3.5?")
3. **Better for Phase 1**: Clear agreement rate between models (90% agreement is meaningful)
4. **LLM-Friendly**: Models are better at reasoning + judgment than precise numerical scoring

**Each Stage Outputs**:
- ✅ **ACCEPT**: Problem/proof meets IMO training quality standards
- ❌ **REJECT**: Problem/proof has critical issues
- 📝 **Detailed Reasoning**: Why the decision was made (for debugging/analysis)

## Model Options

- **gpt-oss-120B**: Internal model, baseline
- **DeepSeek-V3**: Math-specialized, strong in number theory & algebra
- **Claude 3.5 Sonnet**: Excellent logical reasoning (API)
- **GPT-4o**: Strong overall math (API)
