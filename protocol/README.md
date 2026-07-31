# Frozen protocol

The design of this study, fixed before any outcome existed.

Two documents live here because they have no other home: the `docs/` tree that held them was removed
from the repository on 2026-07-30 and is now git-ignored. Both were recovered byte for byte from the
commit that created them, and the git blob hash of each recovered file is recorded below so the
recovery itself is checkable.

The three frozen JSON manifests are **not** duplicated here. They live at the canonical paths the
code loads them from, and a second copy would only create something to drift.

## Documents

| File | Originating commit | Committed | Revisions | git blob | sha256 |
| --- | --- | --- | --: | --- | --- |
| `preregistration_state_dependence.md` | `cef91f4` Preregister BlueDot state-dependence study | 2026-07-27 17:20:52 -0700 | 1 | `a190b4632557f6303c5be817a4b2344935013cc3` | `fac50ab937011f09e7a74a89f4a99cefc4ad600669d2ba1fc53556ef7c62e97b` |
| `execution_decision_tree.md` | `cef91f4`, last content at `9aa527d` | 2026-07-30 09:56:39 -0700 | 8 | `4607f5501fa8e7fd5930c79ca15cd1b990bf78b0` | `9fe078c70f84a2aa36c4e8e6e02764a4737b2dad36f2d684f08a62874abc5a84` |

The **preregistration was committed once and never modified.** Its only two commits are the one that
created it and the one that deleted the `docs/` tree. That is the strongest provenance claim here,
and the one that matters most, because it is the document the hypotheses, target, thresholds,
aggregation unit, and decision rule come from.

The **execution decision tree went through eight revisions**, which is expected and is not a
weakening of the preregistration. It is an operational log of which branch was taken at each gate,
so it necessarily grew as stages completed. It commits to nothing; the preregistration does. The
version here is its final state.

## Frozen manifests

| Artifact | Canonical path | Originating commit | Revisions | Internal content hash |
| --- | --- | --- | --: | --- |
| Prompt split | [`data/prompt_manifests/bluedot_state_dependence_v1.json`](../data/prompt_manifests/bluedot_state_dependence_v1.json) | `c6bfbd7` 2026-07-27 17:50:29 -0700 | 1 | `sha256:bf351c9d73042fcb3d0cdcb18247ded3411413d73000e047f1ba6ba9ab5b25b9` |
| Direction family | [`data/direction_manifests/bluedot_state_dependence_directions_v1.json`](../data/direction_manifests/bluedot_state_dependence_directions_v1.json) | `2e50d4f` 2026-07-27 18:25:17 -0700 | 1 | `sha256:809fbb5b033da740a01574ad5a0ca48f34baca38eef1504a0d66bba8e2fb9138` |
| Calibration plan | [`data/calibration_plans/bluedot_state_dependence_calibration_v1.json`](../data/calibration_plans/bluedot_state_dependence_calibration_v1.json) | `f65b6f9` 2026-07-28 14:27:05 -0700 | 1 | `sha256:a212c6e80f96db55e1aeb0b1879fef441aa6d893a73b7fc43a94142913f97877` |
| Intervention projection | [`data/projections/bluedot_state_dependence_projection_v1.json`](../data/projections/bluedot_state_dependence_projection_v1.json) | `7ca7a87` | 1 | `sha256:0e4207cd69e52567fa703e95811772b3d31dabac3622d1fb1324133bcdad0328` |

Each was committed once and never modified. Each recomputes its own content hash when it loads, so
an edited file fails to parse rather than quietly verifying, and each hash above is the value the
final-test resolution manifest cites. `tests/test_protocol_copies.py` checks these hashes against
the files on disk.

Verify them without loading a model:

```
uv run csf prompts verify --manifest-id bluedot_state_dependence_v1
uv run csf directions verify-family --manifest-id bluedot_state_dependence_directions_v1
uv run csf calibration verify-plan --plan-id bluedot_state_dependence_calibration_v1
```

## Ordering

Every design artifact predates every measurement. The dates above are commit times; the run
timestamps are inside the run manifests.

```
2026-07-27 17:20  preregistration frozen
2026-07-27 17:50  prompt split frozen
2026-07-27 18:25  direction family frozen
2026-07-28 14:27  calibration plan frozen        <- last design decision
2026-07-29 00:26  calibration decision recorded  <- first measurement that selects anything
2026-07-29 18:19  training complete
2026-07-29 18:33  512 forecasts committed
2026-07-30 23:01  final test resolved
```

## Known provenance limitation

The removed `docs/` tree also held a claim-boundaries document, a methodology note, an experiment
log, a compute decision, and a failure-mode catalogue. They are recoverable from git history at
`aac5532^` but are not restored here, because this directory is limited to what fixed the design.
The experiment log was a derived document rebuilt from verified run artifacts; [`REPORT.md`](../REPORT.md)
now serves that role and is likewise written only from verified artifacts.
