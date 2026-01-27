# Quick Start Guide

## What We've Built

A complete high-quality data cleaning pipeline for IMO-level proof problems with:

✅ **5 Quality Assessment Prompts** - Binary decision model (ACCEPT/REJECT)
✅ **2 Processing Scripts** - Decision-based filtering and model comparison
✅ **2 Pipeline Configs** - Ready-to-run experiment and production configs
✅ **Binary Decision Design** - More stable than numeric scores across models

## File Structure

```
rl-data-clean/
├── README.md                           # Complete documentation
├── QUICK_START.md                      # This file
│
├── prompts/                            # LLM prompts (5 files)
│   ├── classify-if-proof.yaml          # Stage 2: Binary classification
│   ├── assess-problem-quality.yaml     # Stage 3: Problem statement quality
│   ├── assess-discussion-quality.yaml  # Stage 4: Discussion quality
│   ├── assess-proof-quality.yaml       # Stage 5: Proof quality ⭐
│   └── assess-imo-readiness.yaml       # Stage 6: IMO-specific evaluation ⭐
│
├── scripts/                            # Python scripts (2 files)
│   ├── postprocess_quality_assessment.py  # Filter by thresholds
│   └── compare_model_outputs.py           # Compare two models (Phase 1)
│
└── configs/                            # Pipeline configurations (2 files)
    ├── experiment-compare-models.yaml  # Phase 1: Small experiment
    └── imo-proof-120b.yaml             # Full pipeline (single model)
```

## Next Steps

### Step 1: Review the Prompts

Take a look at the 5 prompts we designed:

```bash
cd /home/wedu/NeMo-Skills/recipes/rl-data-clean/prompts

# Most important ones to review:
cat assess-proof-quality.yaml      # Core evaluation
cat assess-imo-readiness.yaml      # IMO-specific criteria
```

**Key features:**
- **Binary Decision Model**: Detailed analysis → ACCEPT/REJECT (no numeric scores!)
- **Reasoning First**: LLM analyzes all quality dimensions before making decision
- **Critical Issues Identification**: Explicitly lists deal-breakers
- **Stable Across Models**: Easier agreement than "3 vs 4" scoring

### Step 2: Test a Prompt (Optional)

You can test individual prompts before running the full pipeline:

```bash
# Example: Test proof quality assessment on a sample problem
python -m nemo_skills.inference.generate \
  ++prompt_config=recipes/rl-data-clean/prompts/assess-proof-quality.yaml \
  ++input_file=sample_proof.jsonl \
  ++output_dir=test_output \
  ++model=gpt-oss-120B
```

### Step 3: Prepare Sample Data for Phase 1

Create a small sample (1000 problems) for the experiment:

```bash
# Sample 1000 lines from your raw data
head -n 1000 /path/to/raw_aops_data.jsonl > /workspace/rl-data-clean/experiment/raw_aops_sample_1000.jsonl
```

### Step 4: Run Phase 1 Experiment

Compare two models on the sample:

```bash
# This will run both gpt-oss-120B and deepseek-v3 in parallel
# and compare their outputs

python recipes/rl-data-clean/pipeline/imo_proof_pipeline.py \
  --config experiment-compare-models
```

**Expected output:**
- Agreement rates for each dimension
- List of disagreement cases
- Recommendation: single model vs dual model

**Timeline:** ~5-6 hours for 1000 problems

### Step 5: Analyze Results

```bash
# Review comparison results
cat /workspace/rl-data-clean/experiment/step-4-comparison/proof_quality_comparison.json
cat /workspace/rl-data-clean/experiment/step-6-comparison/imo_readiness_comparison.json
```

**Decision criteria:**
- Agreement > 90% → Use single model (save resources)
- Agreement 80-90% → Use dual model (take intersection)
- Agreement < 80% → Strongly recommend dual model

### Step 6: Run Full Pipeline

Based on Phase 1 results, run the full pipeline:

```bash
# Single model (if agreement >90%)
python recipes/rl-data-clean/pipeline/imo_proof_pipeline.py \
  --config imo-proof-120b

# Or run specific stages only
python recipes/rl-data-clean/pipeline/imo_proof_pipeline.py \
  --config imo-proof-120b \
  --stages extract_problems,classify_if_proof
```

**Timeline:** 2-3 weeks for full dataset (~10k problems)

## Customization

### Adjust Decision Criteria

If pass rates are too high/low, modify the prompts (not numeric thresholds):

**To make STRICTER:**
- Emphasize critical issues in prompt
- Add more examples of what should be REJECTED
- Make decision criteria more explicit

**To make MORE LENIENT:**
- Relax wording around minor issues
- Add examples of borderline cases that ACCEPT
- Adjust "Critical Issues" guidelines

**Why no numeric thresholds?**
- Binary decisions are more stable across models
- Less subjective tuning needed
- Better agreement in Phase 1 experiments

### Add More Models

To test with additional models, edit the experiment config:

```yaml
model_claude:
  model: claude-3.5-sonnet
  server_type: api
  api_key: ${oc.env:ANTHROPIC_API_KEY}

# Then add stages:
assess_proof_quality_claude:
  stage_kwargs: ${model_claude}
```

### Modify Prompts

All prompts are in `prompts/` directory. You can:
- Add more examples
- Adjust rating scales
- Change evaluation criteria
- Add new dimensions

## Common Issues

### Issue 1: Prompt format errors

**Symptom:** Model output doesn't match expected format
**Solution:** Check that output format in prompt matches parser in `postprocess_quality_assessment.py`

### Issue 2: Low pass rates

**Symptom:** Too few problems passing filters
**Solution:**
1. Review sample outputs to see why they're failing
2. Adjust thresholds in config
3. Refine prompts if model is misunderstanding criteria

### Issue 3: Models disagree frequently

**Symptom:** Phase 1 shows <70% agreement
**Solution:**
1. Review disagreement cases manually
2. Determine which model is more accurate
3. Either use the better model alone or use voting with 3 models

## Monitoring Progress

```bash
# Check how many problems pass each stage
for dir in /workspace/rl-data-clean/production/step-*/; do
  echo "$(basename $dir):"
  wc -l $dir/*.jsonl 2>/dev/null | tail -1
done
```

## Questions to Discuss with Team

1. **Binary Decision Design**: Approve the shift from numeric scores to ACCEPT/REJECT?
2. **Model Strategy**: Single model or dual model validation based on Phase 1 results?
3. **Timeline**: Is 5-6 weeks acceptable for full pipeline?
4. **API Models**: Use closed-source models (Claude/GPT-4) or open-source only?
5. **Pass Rate Expectations**: What's acceptable? 30%? 50%? 70% of problems passing all stages?

## Contact

- **Tech Lead**: [Name]
- **Project Doc**: `/home/wedu/imo_proof_pipeline_improvement_plan.md`
- **Branch**: `wedu/rl-data-clean`

---

**Next Meeting Agenda:**
1. Review prompts together
2. Decide on Phase 1 experiment parameters
3. Set timeline and milestones
4. Assign responsibilities
