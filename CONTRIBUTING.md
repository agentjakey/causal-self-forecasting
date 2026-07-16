# Contributing

## The one rule

**Never invent a result.** Not a metric, not a sample count, not a confidence interval, not a
hash, not a screenshot of a number that does not exist yet.

Planned values are labeled planned. Measured values cite a verified run artifact. If you need
a number for a layout, build the empty state instead.

This is enforced by `tests/test_no_fake_results.py`, but the test is a backstop, not the
standard.

## Setup

```powershell
uv sync --extra dev --extra torch
uv run csf doctor
uv run pytest -q
```

Python 3.12. No GPU needed for the test suite.

## Before you push

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q
```

## Code style

* No Unicode in code or output: no em dashes, smart quotes, arrows, or emoji.
* Type hints everywhere. Pydantic for anything that reaches disk.
* `pathlib`, not string paths. Windows-compatible.
* Explicit over implicit. The smallest clear implementation that meets the acceptance
  criteria. No speculative abstraction.
* Comments explain constraints the code cannot state. They do not narrate the next line.

## Adding an experiment

1. If it changes what is being tested, amend `docs/preregistration.md` with a date and a
   reason. Amendments are fine. Silent amendments are not.
2. Fix the split policy before training anything.
3. Add the controls before the result, not after a reviewer asks.
4. Record the outcome in `docs/experiment_log.md`, citing the run id.

## Adding a forecasting method

Every method declares what it may read: prompt, hidden state, adapter identity, clean logits,
intervention vector. The declaration is the experiment, not paperwork.

A prompt-only method that receives any state-derived feature invalidates the headline
comparison. If you are unsure whether a feature leaks, assume it does and ask.

## Adding an intervention mechanism

* Implement it as a pure tensor function in `interventions/tensor_ops.py`, testable with no
  model.
* Decide explicitly what its public view reveals, in `trials/candidates.py`. A mechanism whose
  public label gives away that it is a control is not a control.
* Add its no-op or identity case to the harness controls.

## Tests

* Unit tests for anything that can be tested without weights.
* Integration tests use the fixture model, which is offline and deterministic.
* Mark anything needing real weights with `@pytest.mark.slow`.
* Test the property, not the implementation. `test_capture_matches_the_intervention_point`
  exists because an off-by-one there produces results that look fine and mean something else.

## Claim boundaries

Read `docs/claim_boundaries.md`. A pull request that describes a probe as a self-report, or an
attached forecasting head as introspection, will be asked to change its language regardless of
how good the numbers are.
