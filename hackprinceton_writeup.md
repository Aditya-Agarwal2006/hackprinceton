# Confab: What We Learned By Looking Inside The Model

**HackPrinceton Spring 2026 - Aditya Agarwal & Vaghesan Sundaram**

## What problem we investigated and why

We wanted to know whether an LLM gives away a confabulated answer **inside its
hidden states**, not just in the surface text it prints.

That question matters because a lot of hallucination-detection systems look
strong on paper but are tested on benchmarks with easy shortcuts. If a detector
mostly learns response length, verbosity, or some other superficial cue, then a
high score does not actually tell us much about truthfulness or internal model
behavior.

So the project became two linked questions:

1. Are the common benchmarks actually measuring confabulation?
2. If we remove the shortcuts, is there still a real hidden-state signal left?

The answer ended up being yes, but only after narrowing the claim. The method
works as a **confabulation monitor** under controlled evaluation. It does not
behave like a universal truth detector.

---

## Our approach and methodology

We started by auditing the benchmark.

On raw HaluEval QA, response length alone gets about `0.97` AUC. That means a
detector can look excellent without really understanding anything about
truthfulness. So before trusting any metric, we built **BENCH-2**, a
length-controlled subset that keeps only factual / hallucinated answers for the
same question whose token lengths differ by at most 2 tokens.

With the shortcut controlled, we looked for a cleaner single-pass hidden-state
statistic.

### How we came up with UDC

Our starting point was simple: for each answer token, the model has a hidden
state at every layer. If you subtract consecutive hidden states, you get a
layer-to-layer **update vector**:

`delta^l = h^(l+1) - h^l`

That vector is the direction the model's representation moves at that layer.

The next question was: what should we measure about these vectors?

An early idea was an endpoint-style statistic, which we called TLE. That asked
whether the update norm expands from early layers to late layers. But as we
worked through it, that family turned out to be too endpoint-heavy and less
mechanistically satisfying. It tells you something about the start and end of
the trajectory, but not much about what happens in between.

UDC came from shifting the focus to the thing we actually cared about:

- not just how large the updates are
- but whether **neighboring updates agree on where to go next**

So for each token, we take the cosine similarity between consecutive update
vectors and average them across the layer stack. That gives **Update Direction
Coherence (UDC)**.

In plain English:

- if the model is processing an answer in a stable, internally consistent way,
  neighboring update vectors tend to point in similar directions
- if the model is improvising or internally conflicted, those directions change
  more sharply

For Gemma 4 E2B, the best engineering variant ended up being
`udc_median_tok`, which takes the median token-level UDC across the answer.

### How we interpret the vector directions

This part is important, because the geometry can sound mysterious if we do not
explain it clearly.

Each vector is **not** a claim about truth by itself. It is just the direction
the model's internal representation moves between one layer and the next for a
given token.

The interpretation is local:

- when neighboring update vectors keep pointing in roughly the same direction,
  local coherence is higher
- when they swing apart or reverse, local coherence is lower

That does **not** mean "a smoother line is automatically true." It means:

- for labeled factual examples, the model often shows more coherent local update
  dynamics
- for labeled confabulated examples, the model often shows less coherent local
  update dynamics

The focused geometry view in the app is just a human-readable picture of that
difference. The real claim comes from the benchmark statistics, not from any
single screenshot.

---

## Key findings

### 1. The benchmark shortcut is real

The first major result was that raw HaluEval QA is heavily shortcut-contaminated.
Response length alone reaches about `0.97` AUC.

That means benchmark design is not a side issue. It changes whether a detector
result means anything at all.

### 2. UDC survives after shortcut control

On the controlled BENCH-2 benchmark, UDC remains strong.

Headline Gemma 4 E2B result:

- `udc_median_tok` AUC: **0.7429**
- partial AUC after length control: **0.7363**
- BENCH-2 length baseline: **0.5632**

That is the core scientific result of the project. The signal does not vanish
once the easy shortcut is removed.

### 3. On Gemma, UDC beats TLE

We kept endpoint-style TLE as a comparison feature, but on Gemma it was clearly
weaker than UDC under controlled evaluation.

That was an important design correction. It pushed the project toward the
cleaner, more mechanistic metric.

### 4. TruthfulQA gave us a useful negative result

On TruthfulQA, `udc_median_tok` is near chance:

- TruthfulQA AUC: **0.5008**

This is not automatically suspicious, because TruthfulQA and BENCH-2 are
testing different kinds of wrongness.

- **BENCH-2 / HaluEval-style confabulation**:
  the model produces a plausible-sounding answer that appears inconsistent with
  what it internally "knows"
- **TruthfulQA / misconception-style error**:
  the model may actually carry a false association or false belief from
  training, so there may be less internal conflict in the forward pass

If UDC is measuring internal directional conflict, then it makes sense that it
would work better when the model is internally fighting itself than when it is
smoothly producing a misconception it has already internalized.

That is a narrower claim, but it is a much more defensible one.

### 5. Some transfer ideas did not survive

We also tried to push the method into broader product-style settings, especially
administrative and clinical validation workflows. Those results were weak and
inconsistent.

That dead end mattered. It forced us to stop pretending the method was a
universal validator and sharpen the project around what the data actually
supports.

---

## Problems we faced and the workarounds

### Problem 1: The benchmark looked stronger than it really was

Workaround:

- audit the benchmark first
- build BENCH-2
- headline only the length-controlled results

### Problem 2: The old detector story was too endpoint-heavy

Workaround:

- move away from TLE as the main signal
- center the project on UDC instead

### Problem 3: Raw UDC values are not directly interpretable across models

Workaround:

- fit calibration on labeled BENCH-2 data
- use architecture-specific calibrated verdicts
- keep the headline scientific result in AUC space, not in one arbitrary raw
  threshold

### Problem 4: The geometry visual was too hard to read

Workaround:

- stop showing the whole-answer "spaghetti" view as the main visual
- zoom into a short consecutive token window and short layer slice where the
  coherence gap is strongest
- normalize arrow lengths so the visual shows direction only, which matches what
  UDC actually measures

### Problem 5: The full live model flow was too heavy for a laptop demo

Workaround:

- precompute the Gemma artifacts
- keep the app centered on curated demo cases
- preserve the benchmark evidence in a separate audit tab

That made the final demo much more reliable.

---

## What we would do next

There are three directions that feel most worthwhile from here.

### 1. Broader controlled evaluation

The next serious step is not to add more flashy product claims. It is to test
UDC on more controlled benchmarks and more architectures, especially with the
same discipline about shortcut control.

### 2. Better local explanations

The focused geometry view is already much better than the raw full-trajectory
view, but there is still room to make it more interpretable. The ideal next
step would be a clearer bridge between:

- token-level UDC scores
- local hidden-state direction changes
- the semantic content of the phrase where the model starts to go off-course

### 3. Combining UDC with complementary internal signals

The current project deliberately stayed simple and single-pass. A promising next
step would be testing whether UDC can be paired with other internal-state
features without losing clarity or robustness.

---

## Final takeaway

The biggest lesson from this project is that benchmark rigor and mechanistic
interpretability need to go together.

If the benchmark is shortcut-broken, even a clever detector result can be
misleading. But once the shortcut is removed, hidden-state geometry still carries
a real signal. In our case, that signal is Update Direction Coherence:

- strong enough to survive confound control
- narrow enough to have a real scope boundary
- interpretable enough to turn into a live demo

That is a much better outcome than a flashy "truth detector" claim. It is a
smaller claim, but a real one.

---

## Repo structure

```
hackprinceton/
├── app/                  # Streamlit demo + core UDC modules
│   ├── confab.py         # Main app shell
│   ├── udc_engine.py     # Hidden-state extraction and UDC computation
│   ├── calibration.py    # Calibrated verdict thresholds
│   ├── gemini_client.py  # Gemini API reasoning layer
│   ├── visualization.py  # Plotly charts, gauges, heatmaps
│   ├── scoring.py        # Risk scoring adapter
│   ├── geometry.py       # 3D vector geometry helpers
│   └── demo_data/        # Precomputed fixtures for the demo
├── experiments/          # Benchmark runners and research scripts
├── scripts/              # Calibration, geometry, and bundle-building
├── tests/                # Unit tests
├── notebooks/            # Colab notebook for reproducing Gemma runs
├── docs/                 # Writeup and slide outline
├── COLAB_GPU.md          # Step-by-step GPU reproduction instructions
├── requirements.txt      # Python dependencies
└── PROJECT_STATUS.md     # Current build status
```

## Running the demo app locally

The app uses precomputed Gemma artifacts, since computing it in realtime would be unfeasible for a demo.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
streamlit run app/confab.py
```

To enable the Gemini explanation layer (optional):

```bash
export GEMINI_API_KEY=your_key_here
```

The app has two tabs:

- **Demo**: curated factual vs confabulated examples, token heatmaps, focused
  hidden-state geometry, and Gemini explanations
- **Benchmark Audit**: the benchmark-shortcut story, BENCH-2, headline Gemma
  numbers, and the TruthfulQA scope boundary

## Reproducing the Gemma results

All Gemma experiments were run on Google Colab with a T4/A100 GPU.

See [`COLAB_GPU.md`](COLAB_GPU.md) for the complete step-by-step reproduction
guide, including:

- which files to upload
- how to authenticate for gated Gemma access
- smoke test → BENCH-2 benchmark → calibration → feature sweep → demo case
  generation → 3D geometry precompute
- what each output file is and where it goes

The Colab notebook at `notebooks/gemma_results_colab.ipynb` is a clean
single-notebook entry point that runs the same pipeline.

### Key research scripts

| Script | Purpose |
|---|---|
| `experiments/49_gemma4_trajectory.py` | BENCH-2 + TruthfulQA benchmark run |
| `experiments/50_gemma_feature_sweep.py` | One-pass feature sweep across all metrics |
| `experiments/51_demo_cases.py` | Demo case generation |
| `scripts/calibrate_thresholds.py` | Calibrated verdict thresholds from BENCH-2 |
| `scripts/precompute_demo_geometry.py` | 3D geometry bundle for the app |
| `scripts/build_demo_data_bundle.py` | Package demo data for the app |

Cross-architecture wrappers (included for reference):

- `experiments/52_mistral7b_trajectory.py`
- `experiments/53_qwen25_trajectory.py`

## Tests

```bash
source .venv/bin/activate

# Targeted app sanity suite
pytest tests/test_visualization.py tests/test_confab_audit.py tests/test_live_analysis.py

# Full suite
pytest
```

## Key docs

- [`docs/hackprinceton_writeup.md`](docs/hackprinceton_writeup.md): full blog-style writeup
- [`COLAB_GPU.md`](COLAB_GPU.md): step-by-step GPU reproduction guide
- [`PROJECT_STATUS.md`](PROJECT_STATUS.md): current build status
