# Confab — Project Status Record

This file is the high-level status record for the HackPrinceton build.

It answers four questions quickly:

1. What is this project now?
2. What is already built?
3. What still remains?
4. What should we do next?

It is intentionally shorter and more narrative than [`nextsteps.md`](nextsteps.md).

---

## Project In One Paragraph

Confab is now a **dModel-first interpretability demo** built around a real
research finding: most raw hallucination-detection numbers can be inflated by a
trivial response-length shortcut, but a hidden-state geometry signal called
**UDC** (Update Direction Coherence) still works after length control. On Gemma
4 E2B, the current engineering default metric is `udc_median_tok`, and it
achieves a strong controlled BENCH-2 result while remaining near chance on
TruthfulQA. That gives us a clean scientific story: **UDC is a confabulation
signal, not a universal truth detector**. We also tested broader transfer into
administrative, clinical, and Regeneron-style workflows, and those results were
weak. We are explicitly cutting that branch and centering the project on the
benchmark audit, the mechanistic monitor, the scope boundary, and the live demo.

---

## What Is Built

### Core ML backbone

- UDC engine
- calibration logic
- feature extraction utilities
- Gemma 4 BENCH-2 experiment
- Gemma 4 TruthfulQA scope check
- Gemma feature sweep
- dedicated calibration artifact for the current headline metric

### App-facing layer

- scoring adapter
- risk score computation
- Plotly layer chart
- Plotly risk gauge
- HTML token heatmap
- fixture-backed demo examples
- live-analysis wrapper for Gemma 4
- demo-data bundle for benchmark/audit visuals

### Streamlit shell

- Demo tab
  - curated subject buttons
  - side-by-side factual vs confabulated answers
  - calibrated risk gauges
  - token heatmaps
  - focused local geometry view
  - whole-answer geometry fallback
  - Gemini explanation button
- Benchmark Audit tab
  - headline BENCH-2 number
  - TruthfulQA scope number
  - top-feature ranking
  - benchmark shortcut framing

The app has now been intentionally simplified to these two tabs only. The old
live Analyze flow and the exploratory extra tabs have been removed from the
main demo surface because they were not reliable enough for a laptop-first live
judging setup.

### Gemini tooling

- Gemini client module
  - answer generation
  - claim extraction
  - validation report generation
  - eval dataset generation
- Gemini eval scoring pipeline
- Gemini client tests

### Packaging / reproducibility work already done

- unit tests for the core math, scoring, geometry, visualization, and demo data
- precomputed demo-data bundle inside `app/demo_data/`
- Colab instructions for Gemma GPU runs
- benchmark-number reference doc
- project-status doc and handoff docs
- slide and writeup scaffolds

---

## Current Best Numbers

Current Gemma 4 E2B benchmark picture:

- metric: `udc_median_tok`
- feature-sweep BENCH-2 AUC: `0.7429`
- feature-sweep BENCH-2 partial AUC after length control: `0.7363`
- trajectory-run BENCH-2 `udc_scalar` AUC: `0.7396`
- trajectory-run BENCH-2 length baseline: `0.5632`
- trajectory-run BENCH-2 `tle_scalar` AUC: `0.5981`
- TruthfulQA `udc_median_tok` AUC: `0.5008`

Interpretation:

- good confabulation signal on the controlled benchmark
- near-chance misconception signal on TruthfulQA
- UDC is clearly stronger than TLE on Gemma in the controlled setting

This is the real core of the project.

---

## What We Tested And Rejected

We also tried to push UDC into broader product-style and Regeneron-style tasks.
That did **not** hold up well enough to keep as a primary story.

Representative transfer results:

- scientific claims: mixed and patchy
- clinical / administrative validation prompts: weak
- protocol / contract / source-credibility style examples: often collapsed
  entirely
- some curated one-off examples still separate, but not robustly enough to
  headline as a general workflow validator

Interpretation:

- UDC should **not** be presented as a broad validator for arbitrary
  high-stakes documents
- UDC should **not** be presented as a source-document truth oracle
- the negative transfer result is itself meaningful and should be treated as a
  scientific constraint, not hidden

We are therefore cutting the Regeneron-first and healthcare-first framing from
the main pitch.

---

## What Is Not Finished Yet

The project is not finished, but the remaining work is now much clearer.

Missing or partial pieces:

- final repo-clean packaging for GitHub push
- final slide deck content and visuals
- final demo script / speaking track
- final writeup or deck polish for submission
- Devpost copy, screenshots, and video
- one last README / docs cleanup pass so published instructions match the final app exactly

---

## Current Estimate

- ML backbone: **90-95% done**
- app/demo surface: **90-95% done**
- final dModel submission package: **80-85% done**

This is now mostly a packaging, demo-design, and presentation problem, not a
core-ML problem.

---

## What We Should Do Next

In order:

1. Lock the project as **dModel-first**
2. Freeze the curated demo examples and benchmark numbers
3. Tighten the Benchmark Audit tab copy so it carries the research story clearly
4. Build the clean GitHub-ready package
5. Finish the actual slide deck
6. Write the final judging script
7. Record the demo video and submit on Devpost

The final judging flow should center on:
   - benchmark shortcut problem
   - BENCH-2 fix
   - UDC result
   - TruthfulQA scope boundary
   - focused hidden-state demo

---

## Track Framing

### dModel

This is now the primary and cleanest track fit:

- internal-state probing for confabulation
- benchmark-audit rigor
- hidden-state geometry as the main technical contribution
- explicit scope boundary on TruthfulQA
- negative transfer results treated honestly as part of the finding
- live interactive demo that shows the local update-direction divergence

### MLH / Gemini

Gemini remains useful as an assistive layer:

- drafting candidate answers
- extracting claims from pasted text
- generating short natural-language summaries and reports
- generating evaluation datasets

But Gemini is still secondary. The headline contribution is the hidden-state
monitor plus the benchmark audit; Gemini is the explanation layer that makes
the demo easier to understand.

---

## Important Principle

From this point on, we should optimize for:

- scientific honesty
- a strong dModel demo
- a clean live experience
- clear scope statements
- polished writeup and slides

We should **not** spend the remaining time trying to rescue weak transfer
stories. The strongest version of this project is the one the data actually
supports.
