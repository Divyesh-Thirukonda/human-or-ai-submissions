# Group Number 30 — Human or AI?

Cornell Tech CS 5785, Fall 2026. Required course team name: **Group Number 30**. Actual Kaggle team settings were not changed in this session.

| Team member | NetID | Contribution record |
| --- | --- | --- |
| Divyesh Thirukonda Gowri Shankar | dt579 | Requested and directed the Day 1 investigation; supplied structural hypotheses and required a restart based on the actual competition data. |
| Anusha Ramachandrareddy | To be confirmed | Day 2/3 ownership and work performed must be recorded by the team. |
| Shenye Wang | To be confirmed | Day 2/3 ownership and work performed must be recorded by the team. |

The course assignment sheet confirms these members. Do not treat pending contribution descriptions as work already performed. The course-repository PR URL is still pending.

## Current measured status

The actual-data run has completed ten core candidates across four approaches, tuning three approaches, and all five required learning-curve sizes. The core winner is **character TF-IDF + logistic regression, C=3**:

| Metric | Measured value |
| --- | ---: |
| Training AUROC, 8,000 rows | 0.99914756 |
| Validation AUROC, fixed 2,000 rows | 0.97075600 |
| Source-overlap sensitivity AUROC, 1,992 retained rows | 0.97062737 |
| Validation log loss | 0.28173870 |
| Fit time on the recorded machine | 9.00 seconds |

All four fixed structural ablations completed successfully:

| Structural approach, C=1 | Validation AUROC |
| --- | ---: |
| Style/POS features | 0.906037 |
| POS n-grams | 0.947155 |
| Class-conditional POS trigram surprise | 0.926211 |
| Word TF-IDF + style/POS features | 0.955052 |

After these comparisons, one adaptive extension tested an **equal probability average of character LR C=3 and word/style LR C=1**. It is the current Day 1 handoff recommendation, saved in `results/recommended_config.json`; the core winner remains in `results/selected_config.json` as the fallback and the configuration used for the required learning curve.

| Recommended equal blend | Measured value |
| --- | ---: |
| Training AUROC | 0.998927125 |
| Validation AUROC, 2,000 rows | 0.978162 |
| Improvement over character LR C=3 | +0.007406 |
| Exploratory paired 95% interval for improvement | [+0.00395085, +0.011163475] |
| Source-overlap sensitivity AUROC, 1,992 rows | 0.978040588 |
| Validation log loss | 0.297826494 |
| Fit time on the recorded machine | 40.54 seconds |

The blend improved the competition's ranking metric, but its log loss was worse than the core character model's 0.281739 and it costs more to fit. Its selection rule was recorded before this extension ran, **after** seeing earlier comparisons; this is an adaptive exploratory result, not a wholly prespecified confirmatory test. The 1,000-resample paired interval conditions on the fitted models and does not correct for adaptive selection or correlated sources. The holdout is used for tuning, so none of these values is an independent test estimate or a Kaggle leaderboard score.

Full-data refitting and competition-test prediction are complete. A clean source export and freshly installed pinned environment reproduced all 3,000 probabilities exactly (maximum difference 0). See `results/software_verification.json`. No online Kaggle submission has been made.

## Data access and observed contract

Download the official files from the [competition Data tab](https://www.kaggle.com/competitions/cornell-tech-cs-5785-combined-2026-ai-vs-human-text/data), using your own authorized Kaggle account. Place them in the repository-root `data/` directory, which is `../../data` when working in this team folder:

```text
data/train.csv                 10,000 rows, id,text,label
data/unlabeled.csv              3,000 rows, id,text
data/test.csv                   3,000 rows, id,text
data/sample_submission.csv      3,000 rows, id,label
data/train_starter_2000.csv      2,000 rows, id,text,label
```

`label=0` means human and `label=1` means AI. Submission values must be probabilities of AI authorship, in columns **`id,label`**. The sample's 0.5 values are placeholders. The starter file exactly repeats 2,000 training rows; it is not additional data. The unlabeled file is inspected for integrity on Day 1 but is not used to fit the supervised baselines.

Actual training passages contain 75–350 whitespace words. No exact duplicate passages were found, but moderate lexical overlap and shared openings can reflect related templates. [DATA_NOTES.md](DATA_NOTES.md) records the diligence, relevant limitations, and exploratory training-only feature findings. `results/audit.json` records file hashes and overlap counts. The source-overlap sensitivity diagnostic excludes eight validation IDs sharing either a normalized 80-character opening or word-trigram Jaccard similarity at least 0.50 with a training row; it does not determine model selection.

## Environment

The measured run used **Python 3.12.14**. `requirements.txt` pins the installed dependency versions, including scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.1.1 and NLTK 3.10.4. The recorded machine and package versions are in `results/environment.json`. No GPU or external LLM prediction service is required.

For a portable checkout, start in this team directory:

```bash
python3.12 --version  # use Python 3.12.14
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m nltk.downloader -d ../../data/nltk_data averaged_perceptron_tagger_eng
export NLTK_DATA="$(cd ../../data/nltk_data && pwd)"
.venv/bin/python -m unittest discover -s tests -v
```

The downloaded English POS tagger is a disclosed pretrained annotation model. It is used by the style/POS experiments and the recommended blend; the individual core lexical models work without it. Tagging does not supply authorship labels. Tags are cached within a process, so later bonus runs may have lower tagging costs.

The existing local workspace instead uses a virtual environment at the repository root. From that root, the equivalent commands are:

```bash
.venv/bin/python --version
.venv/bin/python -m pip install -r submissions/group-number-30/requirements.txt
.venv/bin/python -m nltk.downloader -d data/nltk_data averaged_perceptron_tagger_eng
export NLTK_DATA="$PWD/data/nltk_data"
.venv/bin/python -m unittest discover -s submissions/group-number-30/tests -v
```

Use one environment consistently. In the commands below, the working directory is `submissions/group-number-30`, with its portable `.venv`. If using the existing root environment, replace `.venv/bin/python` with `../../.venv/bin/python` after changing into this team directory.

## Reproduce Day 1 on the saved split

Use a fresh results folder to retain the original measurements. Copy the existing split rather than choosing another random seed:

```bash
mkdir -p results_reproduction
cp results/split.json results_reproduction/split.json
cp results/extension_plan.json results_reproduction/extension_plan.json
.venv/bin/python inspect_data.py --data ../../data --results results_reproduction
.venv/bin/python run_day1.py --data ../../data --results results_reproduction --bonus --blend
```

To reproduce only the required core comparison and learning curves, omit `--bonus --blend`. The blend requires the copied `extension_plan.json`; it runs after its character and word/style components. To fit only the blend in an existing reproduction folder, use `--blend-only`. `inspect_data.py` performs inspection without classifier training; its optional `--preview-train 5` prints only training examples. The supplied files use `id`, `text` and `label`; no schema overrides are needed.

There is one editable report: the workspace root's `report/main.tex`, with Q1/Q2 answered directly below the original questions. Its PDF is root `report/output/pdf/course-report-day1.pdf`; Q0 and later-day sections remain unfinished. Edit this LaTeX file directly. Report-generating prose scripts and the duplicate team report/output folders were removed. The course PR contains team code; the report is a separate deliverable.

`run_day1.py` also produces the learning-curve PDF from the recorded fits. To refresh only the canonical report figure without training or changing prose, run:

```bash
.venv/bin/python run_day1.py --results results --plot-only --figure-out ../../report/images/learning_curve.pdf
```

Compile the root `report/main.tex` in a LaTeX editor, or from root `report/` run `pdflatex -interaction=nonstopmode -halt-on-error -jobname=course-report-day1 -output-directory=output/pdf main.tex` twice.

Every controlled run uses seed **20261023**, the same balanced 8,000/2,000 split, and two numerical-library threads. Ordered split IDs and input hashes are persisted. Learned vocabulary, IDF and scaling use training data only. Learning curves use nested stratified subsets of 500, 1,000, 2,000, 5,000 and 8,000 training rows, refitting preprocessing at each size while keeping the selected hyperparameters fixed.

The predeclared core comparison is:

| Approach | Representation | Candidate settings |
| --- | --- | --- |
| Word logistic regression | Word 1–2 grams; minimum document frequency 2; 60,000-feature cap | C = 0.3, 1, 3 |
| Character logistic regression | Character 3–5 grams; minimum document frequency 3; 80,000-feature cap | C = 0.3, 1, 3 |
| Complement Naive Bayes | Same word representation | alpha = 0.1, 1, 3 |
| Calibrated linear SVM | Same word representation | C = 1, three seeded stratified calibration folds, sigmoid |

TF-IDF uses lowercase text, sublinear term frequency, smoothed IDF and L2 normalization. Word tokenization retains tokens of at least two word characters, including stop words. Calibration uses only the outer training partition; internal calibration folds share its vocabulary and IDF. Maximum full-holdout AUROC selects the winner, with first-listed candidates breaking exact ties. Log loss, runtime and source-overlap sensitivity are diagnostics, not alternative selection metrics.

Optional fixed-C=1 ablations compare 28 measured style/POS features, POS 1–3 grams, class-conditional POS trigram surprise, and word TF-IDF combined with style features at fixed weights 1 and 0.1. The class-conditional feature model uses cross-fitted training features. It tests the handwritten sequence idea statistically; it is not a neural next-word model or a grammatical-error detector. Paired bootstrap comparisons use word logistic regression C=1 as the baseline. Their intervals are exploratory, conditional on fitted models, and do not account for training variability, model search, multiple comparisons or correlated sources.

`--resume` is available for interrupted reproduction runs with matching provenance and model files. The checked-in `results/` is a compact evidence set, not a resumable working directory. Original core, curve and bonus records predate resume hardening and ensemble support; their fingerprint intentionally differs from current scoring code. Use a fresh results folder to reproduce them. Resume is an operational convenience, not an independent reproducibility experiment.

## Verify the selected configuration

The following evaluates the recommended blend again on the saved split and saves its validation model under `results/verify/`:

```bash
.venv/bin/python train.py --train ../../data/train.csv --config results/recommended_config.json --results results
```

To verify the core fallback instead, use `results/selected_config.json` and a separate result directory containing the same `split.json`. Choose a fresh result directory containing a copy of `split.json` if a verification run already exists. `train.py --resume` can reuse an existing matching verification run; it does not create an independent refit. The `--model-out` option below is for full-data refitting; validation artifacts use the `verify/` directory.

## Full-data training and Kaggle file generation

These commands completed on all 10,000 labeled examples and produced the verified 3,000-row probability file. A clean-source/fresh-environment rerun matched the output exactly. The `NLTK_DATA` setup above is required for the style/POS component.

```bash
.venv/bin/python train.py --train ../../data/train.csv --config results/recommended_config.json --full-data --model-out artifacts/day1_recommended.joblib
.venv/bin/python predict.py --test ../../data/test.csv --sample-submission ../../data/sample_submission.csv --model artifacts/day1_recommended.joblib --out artifacts/submission_day1_recommended.csv
```

Prediction requires a saved full-data model, checks 3,000 unique template/test IDs, follows template order, writes `id,label`, and verifies probability bounds. Existing model/submission output paths are not overwritten. The core fallback can be refit separately using `results/selected_config.json` and different output names. A later reviewed model can likewise use a separate configuration and output name. These commands do not upload to Kaggle.

## Evidence and artifact layout

| Path | Contents |
| --- | --- |
| `results/split.json` | Fixed ordered training/validation IDs, seed and input fingerprints |
| `results/audit.json` | Consolidated corpus/source-overlap audit, including sample-submission schema and ID order |
| `results/training_feature_audit.json` | Training-only evidence informing the structural hypotheses |
| `results/comparison.json` | All ten core comparison records |
| `results/selected_config.json` | Runnable core winner/fallback configuration |
| `results/learning_curve.json` | All five learning-curve measurements |
| `results/bonus_comparison.json` | Four ablations, failure list and paired bootstrap effects |
| `results/extension_plan.json`, `results/extension_comparison.json` | Adaptive blend plan, measured result and effect versus the core winner |
| `results/recommended_config.json` | Current Day 1 handoff recommendation, separately preserving the core selection |
| `results/validation.predictions.tsv` | Recommended blend's 2,000 validation IDs, labels and probabilities, ready for error analysis |
| `results/day1-evidence.zip` | Complete original small evidence files: individual run records/predictions, subset IDs, plans, plots and earlier report notes; SHA256 manifest included |
| `results/environment.json` | Measured Python, package, platform and thread settings |
| `artifacts/` | Retained full-data model, metadata and ready submission CSV; excluded from Git |

To inspect archived details, extract them explicitly into a separate directory (paths inside the ZIP are relative to the original results folder):

```bash
python -m zipfile -e results/day1-evidence.zip /tmp/group30-day1-evidence
cd /tmp/group30-day1-evidence
shasum -a 256 -c MANIFEST.sha256
```

Use a new extraction directory if that path already exists. The archive contains no competition passages or model weights. Old intermediate models and scratch environments were removed from the workspace; the final full-data model and CSV remain in `artifacts/`. All removed working files also have a verified recovery snapshot beside the private workspace, `../kagglecomp-cleanup-backup-2026-11-05.tar.gz` relative to the repository root. This local backup is not needed to reproduce or submit the project.

The final software check record is `results/software_verification.json`: 18 passing unit tests, clean-source reproduction, output hash, and PDF verification. Small configurations, aggregate metrics, validation-ID probability tables, split IDs and report sources belong in the handoff. Competition CSVs, credentials, model weights, caches and temporary software-test fixtures do not belong in the public course PR. No passage text is needed in the validation prediction tables. Saved model files use joblib and are intended to be regenerated from the pinned code and dependencies.

## Development history, AI assistance and next owners

The fresh actual-data implementation begins at commit [`a4c2805`](https://github.com/Divyesh-Thirukonda/kagglecomp/commit/a4c2805). Subsequent measured results, corrections and conclusions should remain separate meaningful commits. This private workspace does not replace the required course-repository PR.

Codex assisted with data inspection, implementation, tests, experiment execution and report drafting. The user provided the hypotheses and directed the data-first restart. The agent checked outputs and reproducibility interfaces; this does **not** claim independent teammate review or full human verification. Each member must record what they actually inspected, reran, understood or changed for Q6.3. No external LLM API was used to label competition passages, and hidden test labels were not retrieved.

Day 2 still owns the unlabeled-data experiment, controlled improvement decision, final-model recommendation and Q3/Q5/Q6.1. Day 3 still owns generalization/error analysis, five false positives and five false negatives, Q4/Q6.2/Q6.3, final report assembly, course PR and final submission checks. Structural ablations can provide evidence for Day 2; their existence does not automatically complete those sections. Consult `HANDOFF.md` and the root `TEAM_PLAN.md` for the current status and next actions.
