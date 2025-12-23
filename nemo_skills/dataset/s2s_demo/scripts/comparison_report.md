# S2S Demo Evaluation Comparison Report

Comparing 2 model(s):

- duplex-2mim-dec3
- duplex-fp32-nov22

## Metrics Comparison

| Metric | duplex-2mim-dec3 | duplex-fp32-nov22 |
|---|---|---|
| Samples Evaluated | 60 | 60 |
| **Turn-Taking** | | |
|   Latency (ms) ↓ | **299.6** | 404.8 |
|   Precision (%) ↑ | 75.5 | **76.9** |
|   Recall (%) ↑ | **79.5** | 71.8 |
|   F1 (%) ↑ | **76.3** | 72.5 |
| **Barge-In** | | |
|   Success Rate (%) ↑ | **90.0** | 72.3 |
|   Latency (ms) ↓ | **396.9** | 535.8 |
| **Backchanneling** | | |
|   Accuracy (%) ↑ | **0.0** | 0.0 |
| **User Speech (ASR)** | | |
|   WER (%) ↓ | **24.7** | 34.5 |
|   OOB Ratio (words/s) ↓ | **0.000** | 0.009 |
| **Agent Speech (TTS)** | | |
|   WER (%) ↓ | **6.5** | 10.2 |
|   CER (%) ↓ | **2.5** | 4.5 |
|   Hallucination (%) ↓ | **3.0** | 3.8 |
| **LLM Judge (1-5)** | | |
|   Overall Rating ↑ | **3.27** | 3.27 |
|   Full Response ↑ | 3.40 | **3.53** |
|   Sounded Response ↑ | **3.15** | 3.00 |

## Analysis

**duplex-2mim-dec3** is the best model overall based on weighted metrics analysis. Key advantages: duplex-2mim-dec3 leads in Turn-taking F1 (76.3); duplex-fp32-nov22 leads in Turn-taking precision (76.9); duplex-2mim-dec3 leads in Turn-taking recall (79.5); duplex-2mim-dec3 leads in Barge-in success rate (90.0).

---
*↑ = higher is better, ↓ = lower is better, **bold** = best value*
