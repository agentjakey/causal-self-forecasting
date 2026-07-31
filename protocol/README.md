# Frozen protocol

The five artifacts that fixed this study's design before any outcome existed, restored to a tracked
directory so the record survives independently of `docs/`, which was removed from the repository on
2026-07-30 and is now git-ignored.

Nothing here was rewritten, reformatted, or regenerated. Each file was recovered byte for byte from
the commit that created it, and the git blob hash of the recovered file is recorded below so the
recovery itself is checkable.

## Contents and provenance

| File | Originating commit | Committed | Revisions | git blob | sha256 of file |
| --- | --- | --- | --: | --- | --- |
| `preregistration_state_dependence.md` | `cef91f4` Preregister BlueDot state-dependence study | 2026-07-27 17:20:52 -0700 | 1 | `a190b4632557f6303c5be817a4b2344935013cc3` | `fac50ab937011f09e7a74a89f4a99cefc4ad600669d2ba1fc53556ef7c62e97b` |
| `execution_decision_tree.md` | `cef91f4`, last content at `9aa527d` | 2026-07-30 09:56:39 -0700 | 8 | `4607f5501fa8e7fd5930c79ca15cd1b990bf78b0` | `9fe078c70f84a2aa36c4e8e6e02764a4737b2dad36f2d684f08a62874abc5a84` |
| `bluedot_state_dependence_v1.json` | `c6bfbd7` Add deterministic BlueDot prompt manifests | 2026-07-27 17:50:29 -0700 | 1 | tracked at `data/prompt_manifests/` | `0fb579eefdb8a587bb8d9fa4321f4964c3d7c3a29c754be866712fb30eba8646` |
| `bluedot_state_dependence_directions_v1.json` | `2e50d4f` Add deterministic BlueDot direction family | 2026-07-27 18:25:17 -0700 | 1 | tracked at `data/direction_manifests/` | `8d4ef8d810a54da30e5ee3d0c8512b5bf2b8c5cd4be550fe0334250c90084a46` |
| `bluedot_state_dependence_calibration_v1.json` | `f65b6f9` Add BlueDot calibration target and planning | 2026-07-28 14:27:05 -0700 | 1 | tracked at `data/calibration_plans/` | `a65d904910a30d447f75b513e2c7f2897fb61e8e25ad61501e6c02649eaf6f7c` |

The **preregistration was committed once and never modified.** Its only two commits are the one
that created it and the one that deleted the `docs/` tree. That is the strongest provenance claim in
this directory, and it is the one that matters most, because it is the document the hypotheses,
target, thresholds, aggregation unit, and decision rule come from.

The **execution decision tree went through eight revisions**, which is expected and is not a
weakening of the preregistration. It is an operational log of which branch was taken at each gate,
so it necessarily grew as stages completed. It commits to nothing; the preregistration does. The
version here is its final state.

## Internal content hashes

The three JSON artifacts each recompute their own content hash when loaded, and every run manifest
cites those hashes. These are the values the resolution manifest for `bluedot-final-test` records:

| Artifact | Field | Value |
| --- | --- | --- |
| Prompt manifest | `manifest_hash` | `sha256:bf351c9d73042fcb3d0cdcb18247ded3411413d73000e047f1ba6ba9ab5b25b9` |
| Direction family | `family_hash` | `sha256:809fbb5b033da740a01574ad5a0ca48f34baca38eef1504a0d66bba8e2fb9138` |
| Calibration plan | `plan_hash` | `sha256:a212c6e80f96db55e1aeb0b1879fef441aa6d893a73b7fc43a94142913f97877` |

## These are copies, and a test keeps them honest

The code loads these three JSON artifacts from `data/`, not from here. This directory exists so the
protocol reads as one thing rather than three scattered paths, which means there are now two copies
of each file and two copies can drift.

`tests/test_protocol_copies.py` fails if any copy stops being byte-identical to its counterpart
under `data/`. Duplication that is checked by a test is a record; duplication that is not is a
liability.

## Ordering

Every design artifact predates every measurement. The timestamps above are commit times; the run
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

The `docs/` tree also held the claim-boundaries document, the methodology note, the experiment log,
the compute decision, and the failure-mode catalogue. Those are recoverable from git history at
`aac5532^` but are not restored here, because this directory is deliberately limited to the five
artifacts that fixed the design. The experiment log in particular was a derived document, rebuilt
from verified run artifacts; `REPORT.md` now serves that role and is likewise written only from
verified artifacts.
