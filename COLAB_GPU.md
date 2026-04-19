# Colab GPU Instructions for Component 1

These are the exact steps to run the UDC engine on **Gemma 4 E2B** in Google
Colab.

## 1. Start the right runtime

In Colab:

1. Open a new notebook
2. Runtime -> Change runtime type
3. Hardware accelerator -> `GPU`
4. Use at least a `T4` if that is all you have, but `L4` / `A100` is better

## 2. Upload files into one flat Colab directory

You do **not** need to recreate the repo folders in Colab.

Just upload these files directly into the notebook working directory
(``/content``), all at the same level:

- `requirements.txt`
- `udc_engine.py`
- `calibration.py`
- `eval_utils.py`
- `feature_metrics.py`
- `hp_datasets.py`
- `analyze_once.py`
- `calibrate_thresholds.py`
- `49_gemma4_trajectory.py`
- `50_gemma_feature_sweep.py`
- `51_demo_cases.py`

On your machine, these source files live at:

- `requirements.txt`
- `app/udc_engine.py`
- `app/calibration.py`
- `app/eval_utils.py`
- `app/feature_metrics.py`
- `app/hp_datasets.py`
- `scripts/analyze_once.py`
- `scripts/calibrate_thresholds.py`
- `experiments/49_gemma4_trajectory.py`
- `experiments/50_gemma_feature_sweep.py`
- `experiments/51_demo_cases.py`

Also upload `49_gemma4_udc_calibration.json` if you have it (from a previous
BENCH-2 run). The script will find it automatically and use calibrated verdicts.

## 3. Install dependencies

Run:

```python
!python -m pip install --upgrade pip
!pip install -r requirements.txt
```

If you want the strict one-liner instead:

```python
!pip install "torch>=2.6.0" "transformers>=4.57.0" "accelerate>=1.8.0" \
             "numpy>=2.0.0" "safetensors>=0.5.0" "sentencepiece>=0.2.0" \
             "huggingface_hub>=0.30.0" "pytest>=8.3.0"
```

## 4. Authenticate for gated Gemma access if needed

If Gemma 4 is gated for your account:

1. Accept the model license on Hugging Face
2. Create a Hugging Face token
3. In Colab:

```python
import os
os.environ["HF_TOKEN"] = "your_token_here"
```

You can also log in via:

```python
from huggingface_hub import login
login(token="your_token_here")
```

## 5. Run a first Gemma 4 E2B smoke test

Use:

```python
!python analyze_once.py \
  --model google/gemma-4-e2b-it \
  --device cuda \
  --use-chat-template always \
  --prompt "Question: What is the capital of France? Answer:" \
  --response "Paris is the capital of France." \
  --output outputs/gemma4_e2b_smoke.json
```

If that model ID errors, try the alternative capitalization used on parts of
the Hub:

```python
!python analyze_once.py \
  --model google/gemma-4-E2B-it \
  --device cuda \
  --use-chat-template always \
  --prompt "Question: What is the capital of France? Answer:" \
  --response "Paris is the capital of France." \
  --output outputs/gemma4_e2b_smoke.json
```

## 6. What success looks like

The script should print a JSON summary containing:

- `udc_scalar`
- `tle_scalar`
- `num_layers`
- `num_response_tokens`
- `verdict`
- `response_tokens`

This is the key first milestone for Component 1.

## 7. Optional comparison run

Run a plausible confabulation side by side:

```python
!python analyze_once.py \
  --model google/gemma-4-e2b-it \
  --device cuda \
  --use-chat-template always \
  --prompt "Question: What is the capital of France? Answer:" \
  --response "Lyon is the capital of France." \
  --output outputs/gemma4_e2b_wrong.json
```

You are not looking for a magic threshold yet. You are looking for:

- stable execution
- sane token extraction
- non-NaN outputs
- visibly different scores between different answers

## 8. If you want to try the bigger model

Only after E2B works:

```python
!python analyze_once.py \
  --model google/gemma-4-E4B-it \
  --device cuda \
  --use-chat-template always \
  --prompt "Question: What is the capital of France? Answer:" \
  --response "Paris is the capital of France." \
  --output outputs/gemma4_e4b_smoke.json
```

If E4B gives memory trouble, stop there and stay on E2B. That is the correct
hackathon tradeoff.

## 9. Run the actual Gemma benchmark

Once the smoke test works, the important run is not one prompt. It is the
controlled benchmark.

Make sure the flat upload list from Step 2 is present. The important dataset
helper is now `hp_datasets.py`, not `datasets.py`.

Then run a small pilot first:

```python
!python 49_gemma4_trajectory.py \
  --model google/gemma-4-e2b-it \
  --device cuda \
  --use-chat-template always \
  --bench2-max-pairs 25 \
  --truthfulqa-max-pairs 25
```

If that works, run the real version:

```python
!python 49_gemma4_trajectory.py \
  --model google/gemma-4-e2b-it \
  --device cuda \
  --use-chat-template always
```

This writes:

- `outputs/49_gemma4_bench2_rows.csv`
- `outputs/49_gemma4_bench2.json`
- `outputs/49_gemma4_truthfulqa_rows.csv`
- `outputs/49_gemma4_truthfulqa.json`

## 10. Calibrate Gemma verdict thresholds

Do this **after** BENCH-2 finishes:

```python
!python calibrate_thresholds.py \
  --rows outputs/49_gemma4_bench2_rows.csv \
  --metric-col udc_scalar \
  --model google/gemma-4-e2b-it \
  --output outputs/49_gemma4_udc_calibration.json
```

Optional TLE calibration:

```python
!python calibrate_thresholds.py \
  --rows outputs/49_gemma4_bench2_rows.csv \
  --metric-col tle_scalar \
  --model google/gemma-4-e2b-it \
  --output outputs/49_gemma4_tle_calibration.json
```

## 11. The actual decision rule

After the benchmark run, compare:

- BENCH-2 AUC for `udc_scalar`
- BENCH-2 AUC for `tle_scalar`
- TruthfulQA scope result for both

That tells you:

- whether UDC remains the primary signal on Gemma
- whether TLE is stronger on Gemma
- whether the best app story is UDC-only or UDC+TLE together

Do **not** decide this from a single smoke-test verdict.

---

## 12. Run the demo case finder — Step 1: generate factual answers

This generates and scores demo examples for History, Science, English, and Math.
Uses Gemma for BOTH generation and scoring — same model for both steps.

```python
!python 51_demo_cases.py \
  --model google/gemma-4-e2b-it \
  --device cuda
```

Expected runtime: ~15–25 minutes on an A100 (3 questions × 4 subjects × 2 answers
+ UDC scoring for each).

Outputs:
- `outputs/51_demo_cases_raw.json` — all 12 scored candidates
- `outputs/51_demo_cases_best.json` — best pair per subject (the app fixture)

The summary table printed at the end looks like:

---

## 13. Precompute the 3D geometry bundle for the 4 demo subjects

Once you are happy with `demo_cases.json`, precompute the real update-vector
geometry that the app will render instantly during the demo.

### Upload these files flat into `/content`

- `requirements.txt`
- `udc_engine.py`
- `geometry.py`
- `demo_cases.json`
- `precompute_demo_geometry.py`

On your machine those live at:

- `hackprinceton/requirements.txt`
- `hackprinceton/app/udc_engine.py`
- `hackprinceton/app/geometry.py`
- `hackprinceton/app/demo_data/demo_cases.json`
- `hackprinceton/scripts/precompute_demo_geometry.py`

### Run

```python
!python precompute_demo_geometry.py \
  --model google/gemma-4-e2b-it \
  --device cuda \
  --demo-cases demo_cases.json \
  --output demo_geometry.json
```

### What it writes

- `demo_geometry.json`

This file contains, for each of the 4 demo subjects:

- factual-answer 3D geometry
- confabulated-answer 3D geometry
- PCA variance info
- token trajectories through the projected layer-update space

### What to do after the run

Move `demo_geometry.json` back into:

- `hackprinceton/app/demo_data/demo_geometry.json`

Then the Streamlit app will render the 3D views instantly on the Demo tab
without trying to load Gemma locally just to build the geometry.

```
Subject       Gap  Factual UDC  Confab UDC  Question (truncated)
------------------------------------------------------------------------
History    +0.042       -0.118      -0.076  What were the underlying causes...
Science    +0.037       -0.103      -0.066  How does photosynthesis convert...
English    +0.051       -0.125      -0.074  What are the central themes of...
Math       +0.029       -0.097      -0.068  What does Euler's identity expr...
```

**After the run:** keep `outputs/51_demo_cases_best.json` in Colab for Step 2.
Also upload it to `/content` for the next script.

**NOTE:** Gemma's RLHF training will likely prevent real confabulation even with
the confab system prompt — the "confabulated" answers will just be correct
paraphrases. That is expected. Step 2 fixes this.

---

## 13. Run the demo case scorer — Step 2: short-format pairs

Long educational explanations (~100 tokens) don't work with the BENCH-2
calibration — all scores land in FAIL territory because the per-token signal
is diluted. BENCH-2 answers average 5–6 tokens. Step 2 uses very short
1-sentence answers (15–30 tokens) that match the calibration's training range.

Both factual and confabulated answers are hard-coded in the script. Confabulated
answers contain multiple specific errors per sentence (wrong names, wrong years,
wrong processes) so most response tokens are processing incorrect information.

Upload to `/content` (flat):
- `udc_engine.py`, `calibration.py`, `eval_utils.py`, `feature_metrics.py`
- `49_gemma4_udc_calibration.json`
- `51c_short_format.py`

Run:

```python
!python 51c_short_format.py \
  --model google/gemma-4-e2b-it \
  --device cuda
```

Expected runtime: ~3–5 minutes (scoring only, 8 total forward passes).

All pairs redesigned to use **categorical/conceptual errors** (not named-entity
swaps). Named-entity swaps (different person, different city) barely move UDC
because the surrounding sentence structure is identical. Categorical errors
(wrong genre, wrong outcome, wrong mathematical operation) produce stronger
internal conflict in Gemma's hidden states.

Injected errors per subject:
- **History:** Confederacy won (not Union) — opposite outcome; wrong year; wrong
  social consequence. Every downstream claim inverts.
- **Science:** nucleus (not mitochondria) — wrong organelle category; ADP not
  ATP; photosynthesis not respiration. All four claims wrong.
- **English:** Hamlet is a comedy (not tragedy) — categorical genre error;
  celebration not revenge; coronation not murder. Opposite emotional register.
- **Math:** derivative measures area (describes the integral, not the derivative)
  — wrong mathematical operation, categorical inverse.

Run (first attempt — use BENCH-2 calibration):

```python
!python 51c_short_format.py \
  --model google/gemma-4-e2b-it \
  --device cuda
```

**If BENCH-2 calibration still doesn't produce PASS vs FAIL verdicts:**

```python
!python 51c_short_format.py \
  --model google/gemma-4-e2b-it \
  --device cuda \
  --demo-calibrate
```

The `--demo-calibrate` flag fits a calibration from the 8 demo examples
(4 factual + 4 confab, which we have ground-truth labels for). This is honest:
the demo uses demo-calibrated thresholds; the benchmark tab still uses BENCH-2.
It also saves `outputs/demo_calibration.json` alongside the fixture.

**After the run:** download both `outputs/51c_short_scored.json` and
(if used) `outputs/demo_calibration.json` → commit as
`app/demo_data/demo_cases.json` and `app/demo_data/demo_calibration.json`.
