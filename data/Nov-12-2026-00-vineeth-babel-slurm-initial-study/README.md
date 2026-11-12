# Data for the Nov 12, 2026 study: auxiliary math practice during GRPO training of Qwen3-1.7B

Frozen. These are the problem sets, labels and manifests the study used. What each set is for is explained in
[`Nov-12-2026-understanding-experiments.md`](../../reports/Nov-12-2026-00-vineeth-babel-slurm-initial-study/Nov-12-2026-understanding-experiments.md) §2.

| Path | Contents |
|---|---|
| `main_v1_manifest.json` | the training pools (`hard` 46, `related` 63, `unrelated` 63) and the probes (`dev` 53, `probe_J_H` 32, `probe_J_R` 28, `probe_J_U` 32) |
| `panels/panels.json` | problem IDs of dev, the sealed test (128) and the probes; `panels.sha256` holds their hashes |
| `screening/` | base-model success counts used to select hard and practice problems; `match_k8.json` pairs R with U problems |
| `authored/` | practice problems written for the study (`R1–R5` factoring, `U1–U3` modular arithmetic) |
| `annotations/` | annotator labels (two passes plus a third read on disagreement): is factoring central, is it a direct exercise, is modular arithmetic used |
| `external/` | pinned copies of source problems (MATH under `a17/`, an Omni-MATH tranche) |
| `stress/` | Harder (95) and SuperHard (126) transfer sets; `superseded-d143/` is an earlier version kept for the record |
| `control/` | 64 MATH test problems from subjects no arm practised |
| `cue/` | 24 problems with and without a one-sentence factoring hint |

Problem IDs look like `intermediate_algebra/train/00481` (MATH subject/split/row). `syn-R…` and `syn-U…` are authored
problems. Some older scripts in `src/qis/data/` and `src/qis/analysis/` still open paths as `data/…`. These are the
pre-move paths: run them with this folder as `data/`, or pass the paths explicitly.
