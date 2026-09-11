# HQKM reproducibility package

Reference implementation and result artifacts for *"Can Hybrid
Quantum–Classical Algorithms Accelerate Machine-Learning Tasks? A Cost–Benefit
Benchmark of Budgeted Quantum Kernel Acquisition"*. The experiments are
classically simulated; no quantum-hardware or quantum-advantage claim is made.

**Release status:** audited reproducibility candidate `v1.0.0`. All 34
regression tests pass. Before publishing, replace the `REPLACE_WITH_*` metadata
tokens and run `python scripts/release_check.py`.

The DOI is assigned only after the repository owner publishes a GitHub release
and the connected Zenodo record completes archival.

**Post-audit correction.** The current code uses split-first, train-fitted
preprocessing for real data; finite-shot sampling is applied whenever a model
reports finite-shot circuit cost; and symmetric landmark blocks are sampled
once per unique entry. Result files dated before this correction are legacy
artifacts and must not be reported as results of the corrected implementation.
Run a new paper profile after the regression suite passes.

## What this is

A classical-simulable testbed for one question: *given a dictionary of
classical kernels and one quantum kernel, when is the quantum kernel worth its
circuit-execution cost?* Three mechanisms:

1. **Exploratory diagnostic** `Φ = ΔA / D` — alignment gained by admitting the
   quantum kernel, divided by an effective-dimension complexity price with a
   `1/n` floor. It is evaluated empirically and is not a population-risk
   certificate.
2. **Emulation screen** — least-squares emulation of the quantum *landmark*
   Gram by the classical dictionary; if `R² ≥ 0.99`, the quantum kernel is
   declined after the initial landmark cost but before remaining columns are
   measured.
3. **Noise-aware Nyström estimator** — sketch the quantum Gram on `m` landmarks
   with `s` shots, truncate eigenvalues below `τ_trunc = τ_c·√(m/(4s))`, repair
   PSD. Circuit executions drop from `s·N(N+1)/2` to `s·(Nm + m(m+1)/2)`.

## Install

```bash
git clone https://github.com/REPLACE_WITH_OWNER/REPLACE_WITH_REPOSITORY.git
cd REPLACE_WITH_REPOSITORY
python -m pip install -e ".[all]"
python -m pytest -q
python run_experiments.py --list-datasets
```

Leave-one-dataset-out utility-gate validation and the accuracy--circuit-cost
Pareto analysis can be reproduced with:

```bash
python run_loodo.py --out results/loodo_pareto_12tasks_5seeds.json
python plot_loodo_pareto.py results/loodo_pareto_12tasks_5seeds.json \
  results/loodo_pareto_frontier.png
```

The bundled suite contains 12 binary tasks from four independent source
datasets. The output therefore reports both task-level LOODO and the stricter
leave-one-source-group-out analysis to expose possible pseudo-replication.

Sequential budgeted landmark acquisition can be smoke-tested with:

```bash
python run_budgeted_acquisition.py \
  --out results/budgeted_acquisition_smoke.json
```

The `budgeted_hybrid_mkl` model ranks candidate landmarks from training
residuals, buys only new quantum columns, selects its stopping round on the
tuning split, and touches the test split only after the path is frozen. Quantum
exploration costs remain charged even when the final decision falls back to the
classical model.

The leakage-safe multi-objective extension combines residual, uncertainty and
greedy diversity ranks, imposes a hard fit-execution budget, and requires
non-negative gain per million executions. Its development-only screen is:

```bash
python run_development_multiobjective.py --root /path/to/development/csvs \
  --out results/development_multiobjective_screen.json
```

That script rejects every dataset outside its seven-name development allowlist.
After configuration freeze, the one-shot external command is:

```bash
python run_locked_multiobjective_external.py --root /path/to/external/files \
  --out results/locked_multiobjective_external.json
```

The external script embeds the selected configuration and records its SHA-256
digest. External outcomes must not be used to retune this configuration.

The fully classical pre-acquisition meta-gate is evaluated with strict
leave-one-dataset-out folds:

```bash
python analyze_pre_acquisition_gate.py \
  --input results/development_multiobjective_screen.json \
  --out results/pre_acquisition_gate_loodo.json
```

Its interface accepts only ten allowlisted train/tune meta-features. The current
seven-dataset LODO result is a negative control: it saves cost but fails to
identify beneficial acquisitions reliably, so it must not be promoted to a new
external evaluation without additional meta-training datasets.

Uploaded raw CSV benchmarks can be audited and run without network access:

```bash
python run_uploaded_benchmarks.py --root /path/to/csv_directory \
  --mode classical --out results/uploaded_classical_pilot.json
python run_uploaded_benchmarks.py --root /path/to/csv_directory \
  --mode compare --shots 100 --out results/uploaded_budgeted_compare.json
```

The accepted/rejected schemas are explicit in `hqkm/local_datasets.py`.
Headerless Adult and Ionosphere files preserve their first rows, and duplicate
feature vectors are handled before splitting to prevent cross-split identity
leakage.

The locked external ablation and its ten-source aggregation are reproduced by:

```bash
python run_external_ablation.py --root /path/to/external_csvs \
  --out results/external_v3_locked_ablation.json
python analyze_external_ablation.py results/external_v3_locked_ablation.json \
  --development results/uploaded_budgeted_compare.json \
  results/uploaded_v2_new_budgeted_compare.json \
  --out results/external_v3_analysis_and_10source.json
python plot_external_pareto.py results/external_v3_analysis_and_10source.json \
  results/external_v3_pareto.png
```

NumPy, pandas, matplotlib and SciPy are the runtime dependencies.
scikit-learn is optional for bundled/OpenML benchmark loading; pytest is an
optional test dependency. See `pyproject.toml` and `environment.yml`.

## Datasets (auto-download on first use)

| name | kind | N | note |
|---|---|---|---|
| `S1_mixed` … `S5_noise` | synthetic | you choose | controlled families: mixing weight sweeps classical→quantum structure; in-RKHS; entangled anchor; high-frequency (must decline); pure noise (must decline) |
| `breast_cancer`, `digits01`, `wine` | real, bundled | 0.1–0.6k | ship with scikit-learn, no network |
| `adult` (OpenML 1590), `electricity` (151), `magic` (1120) | real, OpenML | 19–49k | downloaded once to `~/.hqkm_data` (override with `HQKM_DATA_DIR`), then cached |
| `miniboone` (41150) | real, OpenML | 130k | subsampled to the requested N |

## Models (the baselines, one interface)

| name | role |
|---|---|
| `classical_single` | best single tuned classical kernel |
| `classical_mkl` | **primary baseline**: identical code path to ours with β_q ≡ 0 |
| `quantum_only` | quantum kernel alone, exact or sketched |
| `quantum_aligned` | SOTA contrast: encoding scale trained by alignment maximization |
| `qnn` | SOTA contrast: variational circuit, SPSA-trained |
| `emulator_prior_art` | closest prior art, reimplemented: Nyström + k-means landmarks + classical surrogate, no criterion, no PSD repair |
| `hybrid_mkl` | evaluated hybrid method |

Fairness is structural: same ridge solver, same λ grid, same sketch budget,
same stratified train/tune/test splits per seed. λ, m, s are tuned on the tune
split only; the criterion threshold τ is never tuned on data. Any number for
the prior-art pipeline is measured by us, never copied from its paper.

## Run

```bash
python run_experiments.py --profile smoke                     # ~1 min
python run_experiments.py --profile pilot                     # ~hours
python run_experiments.py --profile paper --out results.json  # the paper run
python run_experiments.py --profile paper --datasets adult magic --studies E4
python -c "from hqkm.figures import make_figures; make_figures('results.json')"
```

Studies: **E1** diagnostic association (Φ vs realized Δaccuracy, confusion matrix,
rank correlation) · **E2** estimator quality over (m, s): ε_Fro, one- and
two-sided d_eff drift, rank, τ_trunc · **E3** cost scaling vs N · **E4** full
benchmark, paired per seed, Wilcoxon + paired-t + bootstrap CI + Holm–Bonferroni
per dataset · **E5** gate study: fire rate, gate-error rate, executions saved.

## Layout

```
hqkm/
  quantum.py    product & entangled feature maps, shot noise, <Z0> observable
  kernels.py    classical dictionary, MKL alignment solver, KRR
  estimator.py  Nyström sketch, τ_trunc rule, PSD repair, execution accounting
  criterion.py  Φ, redundancy certificate, bound gap
  baselines.py  the seven models behind one interface
  datasets.py   synthetic families + auto-downloading real benchmarks
  metrics.py    accuracy/AUC/F1, paired tests, Holm–Bonferroni
  protocol.py   E1–E5
  figures.py    paper figures from results.json
run_experiments.py    CLI (profiles: smoke / pilot / paper)
tests/                unit tests (run: python -m pytest tests)
```

## Reproducibility, citation, and DOI

- Start with `REPRODUCIBILITY.md` for the clean-environment workflow.
- See `DATA.md` for dataset access and redistribution constraints.
- Citation metadata are in `CITATION.cff`; GitHub displays them through its
  **Cite this repository** interface after publication.
- Follow `DOI_RELEASE_GUIDE_FA.md` to connect the public repository to Zenodo
  and mint the DOI from release `v1.0.0`.
- `python scripts/release_check.py` blocks release while author/repository
  placeholders or machine-specific paths remain.

## Reproducibility and honesty notes

- Results carry seeds and full config; `results.json` is the artifact.
- Known open items (documented, not hidden): the analytic value of τ is open
  (τ=0 is a proxy); the alignment→excess-risk link is not established; the PSD
  repair was a measured no-op in pilots (reported via `negative_mass`).
- Synthetic families exist because on real data the quantum effective dimension
  cannot be varied independently; they are not evidence of practical advantage.

## License

MIT (see `LICENSE`). Datasets keep their upstream licenses (CC BY 4.0 for the
OpenML/UCI sets used here).
