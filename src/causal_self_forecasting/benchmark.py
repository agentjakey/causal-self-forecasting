"""Systems benchmark for a real pretrained model.

This answers three operational questions before any compute is bought:

1. Do the pinned weights load here at all?
2. What does one forward pass cost on this machine?
3. Does the hook-owned capture path behave correctly on real weights, not just on the
   fixture?

Question 3 is the one that matters most. The transformers 5 hidden-state indexing problem
(see `models/capture.py`) was found on the fixture, and the fix has to hold on the real
architecture too. So this benchmark does not merely check that a hook fires. It patches the
residual stream at a layer with a known vector and reads that layer back, requiring the two
to match. A hook can fire and still be read from the wrong point, and that is exactly the bug
that produces plausible, meaningless numbers.

What this is not: a CSF-Bench result. There is no model organism, no estimated direction, and
no forecaster involved. The accuracy measured here is a handful of clean multiple-choice
items, which checks that scoring is wired up and nothing more. Every artifact this module
writes is typed so that it cannot claim otherwise.
"""

from __future__ import annotations

import ctypes
import math
import os
import platform
import random
import shutil
import statistics
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig, TaskConfig, config_hash
from .hashing import atomic_write_json, hash_file, write_jsonl
from .interventions.tensor_ops import InterventionPayload
from .logging_utils import info, warn
from .models.capture import capture_hidden_states, run_with_intervention, validate_layer
from .models.loader import LoadedModel, ModelLoadError, load_model
from .models.scoring import resolve_label_token_ids, score_logits
from .paths import (
    ARTIFACT_HASHES,
    BENCHMARK,
    BENCHMARK_ITEMS,
    ENVIRONMENT,
    RUN_MANIFEST,
    private_dir,
    run_dir,
)
from .reproducibility import derive_seed, environment_snapshot, set_global_seed
from .schemas import (
    AccessStatus,
    ArtifactHashRecord,
    BenchmarkCapture,
    BenchmarkClassification,
    BenchmarkEvaluation,
    BenchmarkMemory,
    BenchmarkModelInfo,
    BenchmarkRecord,
    BenchmarkTaskInfo,
    BenchmarkTiming,
    ComputeEstimate,
    InterventionSpec,
    Mechanism,
    PromptVariant,
    RunManifest,
    Split,
)
from .tasks.loader import TaskLoadError, load_prepared_task, task_manifest_path

# The scoring format. The existing trial pipeline feeds `PromptVariant.prompt_text` straight
# to the model, with no chat template, and scores the next token after "Answer:". The
# benchmark deliberately matches it: a benchmark that measured a different prompt format from
# the one the experiments use would answer a question nobody asked.
SCORING_FORMAT = "raw_completion_next_token_after_answer_colon"

# Headroom over the reported weight size, covering the incomplete-download temp files.
_DISK_HEADROOM = 1.25


class BenchmarkError(RuntimeError):
    """Raised when the benchmark cannot proceed.

    Carries an `AccessStatus` so the CLI can report *why* in a way that points at the right
    remedy. "Model load failed" when the real problem is an unaccepted license sends someone
    to debug entirely the wrong thing.
    """

    def __init__(self, status: AccessStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class AccessCheck:
    """The outcome of looking at the weights without downloading them."""

    status: AccessStatus
    detail: str
    resolved_sha: str | None = None
    cached: bool = False
    required_bytes: int | None = None
    free_bytes: int | None = None


# ---------------------------------------------------------------------------
# Process memory
# ---------------------------------------------------------------------------


class _WindowsMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def read_process_memory() -> tuple[int | None, int | None, str]:
    """Return (current rss, peak rss, how it was measured).

    Uses the OS rather than a new dependency: PSAPI on Windows, /proc on Linux. Anywhere
    else, both values are None and the reason is recorded. A null is the honest answer; an
    estimate dressed up as a measurement is not.
    """
    system = platform.system()
    if system == "Windows":
        try:
            counters = _WindowsMemoryCounters()
            counters.cb = ctypes.sizeof(_WindowsMemoryCounters)
            handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
                handle, ctypes.byref(counters), counters.cb
            )
            if not ok:
                return None, None, "windows psapi GetProcessMemoryInfo returned failure"
            return (
                int(counters.WorkingSetSize),
                int(counters.PeakWorkingSetSize),
                "windows psapi working set",
            )
        except (OSError, AttributeError) as error:
            return None, None, f"windows psapi unavailable: {type(error).__name__}"

    if system == "Linux":
        try:
            values: dict[str, int] = {}
            for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, raw = line.split(":", 1)
                    values[key] = int(raw.strip().split()[0]) * 1024
            return values.get("VmRSS"), values.get("VmHWM"), "linux /proc/self/status"
        except (OSError, ValueError, IndexError) as error:
            return None, None, f"/proc/self/status unreadable: {type(error).__name__}"

    return None, None, f"no supported memory measurement on {system}"


# ---------------------------------------------------------------------------
# Access checking
# ---------------------------------------------------------------------------


def _cache_dir() -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE)


def _weights_cached(model_id: str, revision: str) -> bool:
    """True when both the config and the weights are already in the local cache."""
    from huggingface_hub import try_to_load_from_cache

    for filename in ("config.json", "model.safetensors"):
        hit = try_to_load_from_cache(repo_id=model_id, filename=filename, revision=revision)
        if not isinstance(hit, str):
            return False
    return True


_METADATA_ATTEMPTS = 3
_METADATA_BACKOFF_SECONDS = 2.0


def _model_info_with_retry(api: Any, config: ModelConfig) -> Any:
    """Fetch model metadata, retrying only transport failures.

    The probe is one small request, and a dropped connection on it is not evidence about
    access. Retried because this was observed failing and then succeeding immediately
    afterwards, and reporting "network failure" for a blip would send someone to debug a
    working network.

    Only transport errors are retried. A gated repo or a missing revision is an answer, not
    a blip, and retrying it would just be slow.
    """
    import httpx

    last: Exception | None = None
    for attempt in range(1, _METADATA_ATTEMPTS + 1):
        try:
            return api.model_info(
                config.model_id, revision=config.revision, files_metadata=True, timeout=30
            )
        except (httpx.HTTPError, ConnectionError, TimeoutError) as error:
            last = error
            if attempt < _METADATA_ATTEMPTS:
                warn(
                    "hub metadata request failed, retrying",
                    attempt=attempt,
                    of=_METADATA_ATTEMPTS,
                    error=type(error).__name__,
                )
                time.sleep(_METADATA_BACKOFF_SECONDS * attempt)
    raise last if last is not None else RuntimeError("model metadata retry loop fell through")


def check_model_access(config: ModelConfig, offline: bool) -> AccessCheck:
    """Determine whether the pinned weights can be reached, without downloading them.

    Deliberately done before loading, so that a missing license or a missing network is
    reported as itself instead of surfacing later as an opaque loader traceback.
    """
    if config.kind == "fixture":
        return AccessCheck(
            status=AccessStatus.FIXTURE_LOCAL,
            detail="fixture model is built locally; no hub access required",
        )

    import httpx
    from huggingface_hub import HfApi, get_token
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    cached = _weights_cached(config.model_id, config.revision)

    if offline:
        if cached:
            return AccessCheck(
                status=AccessStatus.CACHED_LOAD,
                detail="offline mode; weights found in the local cache",
                resolved_sha=config.revision,
                cached=True,
            )
        return AccessCheck(
            status=AccessStatus.OFFLINE_CACHE_MISS,
            detail=(
                f"offline mode was requested but {config.model_id} at revision "
                f"{config.revision} is not in the local cache at {_cache_dir()}. "
                "Run once without --offline to populate it."
            ),
        )

    # Presence only. The token's value is never read into a variable that could be logged,
    # persisted, or included in an artifact.
    token_present = get_token() is not None

    try:
        details = _model_info_with_retry(HfApi(), config)
    except GatedRepoError:
        # Must precede RepositoryNotFoundError: GatedRepoError is a subclass of it.
        if not token_present:
            return AccessCheck(
                status=AccessStatus.NO_AUTHENTICATION,
                detail=(
                    f"{config.model_id} is gated and no Hugging Face credential was found. "
                    "Authenticate with `uv run hf auth login` using a read token."
                ),
            )
        return AccessCheck(
            status=AccessStatus.GATED_ACCESS_DENIED,
            detail=(
                f"authenticated, but this account has not been granted access to "
                f"{config.model_id}. Accept the license on "
                f"https://huggingface.co/{config.model_id} and retry."
            ),
        )
    except RevisionNotFoundError:
        return AccessCheck(
            status=AccessStatus.REVISION_UNAVAILABLE,
            detail=(
                f"revision {config.revision} does not exist in {config.model_id}. "
                "The pinned revision is part of the experiment's provenance and is not "
                "substituted automatically."
            ),
        )
    except RepositoryNotFoundError:
        # The hub returns 401 for a gated repo when unauthenticated, so an unauthenticated
        # caller cannot distinguish "missing" from "not allowed to know".
        if not token_present:
            return AccessCheck(
                status=AccessStatus.NO_AUTHENTICATION,
                detail=(
                    f"{config.model_id} could not be read and no Hugging Face credential was "
                    "found. Authenticate with `uv run hf auth login` using a read token."
                ),
            )
        return AccessCheck(
            status=AccessStatus.GATED_ACCESS_DENIED,
            detail=(
                f"{config.model_id} is not visible to this account. If it is gated, accept "
                f"the license on https://huggingface.co/{config.model_id}."
            ),
        )
    except (HfHubHTTPError, httpx.HTTPError, OSError) as error:
        # httpx.HTTPError is listed explicitly because it is not an OSError. huggingface_hub
        # 1.x moved from requests to httpx, so a dropped connection arrives as
        # httpx.ConnectError and an OSError-only clause lets it escape as a raw traceback
        # instead of being reported as the network failure it is.
        return AccessCheck(
            status=AccessStatus.NETWORK_FAILURE,
            detail=f"could not reach the Hugging Face hub: {type(error).__name__}: {error}",
        )

    if details.sha != config.revision:
        return AccessCheck(
            status=AccessStatus.REVISION_UNAVAILABLE,
            detail=(
                f"requested revision {config.revision} resolved to {details.sha}; refusing to "
                "continue because the run would not be the pinned one"
            ),
        )

    required = _required_bytes(details)
    free = shutil.disk_usage(_cache_dir().anchor or "/").free

    if not cached and required is not None and free < required * _DISK_HEADROOM:
        return AccessCheck(
            status=AccessStatus.INSUFFICIENT_DISK_SPACE,
            detail=(
                f"downloading {config.model_id} needs about {required / 1e9:.1f} GB plus "
                f"headroom, but only {free / 1e9:.1f} GB is free on {_cache_dir().anchor}"
            ),
            resolved_sha=details.sha,
            required_bytes=required,
            free_bytes=free,
        )

    return AccessCheck(
        status=AccessStatus.CACHED_LOAD if cached else AccessStatus.REMOTE_DOWNLOAD,
        detail=(
            "weights are in the local cache"
            if cached
            else f"weights will be downloaded (about {(required or 0) / 1e9:.1f} GB)"
        ),
        resolved_sha=details.sha,
        cached=cached,
        required_bytes=required,
        free_bytes=free,
    )


def _required_bytes(details: Any) -> int | None:
    """Sum the weight-file sizes the hub reports, or None if it did not report them."""
    siblings = getattr(details, "siblings", None) or []
    total = 0
    seen = False
    for sibling in siblings:
        name = getattr(sibling, "rfilename", "") or ""
        size = getattr(sibling, "size", None)
        if name.endswith((".safetensors", ".bin")) and isinstance(size, int):
            total += size
            seen = True
    return total if seen else None


def _is_network_error(error: BaseException) -> bool:
    """Walk an exception chain looking for a transport failure.

    The loader wraps everything as a ModelLoadError, so a connection dropped part way
    through a multi-gigabyte download would otherwise be reported as "the model failed to
    load", sending someone to debug their weights when they should just retry.
    """
    import httpx

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, httpx.HTTPError | ConnectionError | TimeoutError):
            return True
        current = current.__cause__ or current.__context__
    return False


@contextmanager
def _offline_env(offline: bool) -> Iterator[None]:
    """Force the hub into offline mode for the duration of a load.

    Set through the environment rather than a loader argument so that the centralized loader
    stays untouched and every path underneath it (tokenizer, config, weights) obeys the same
    rule.
    """
    if not offline:
        yield
        return
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous


# ---------------------------------------------------------------------------
# Item selection
# ---------------------------------------------------------------------------


def select_variants(
    variants: list[PromptVariant],
    split: Split,
    max_items: int,
    seed: int,
    task_name: str,
) -> list[PromptVariant]:
    """Pick prompts deterministically from one split.

    Subsampled with a seeded shuffle rather than by truncating the sorted list, for the same
    reason trial generation does it: truncation takes every wrapper variant of the first
    question, so a small run silently covers one item under eight framings instead of a
    spread of the split.
    """
    eligible = sorted(
        (variant for variant in variants if variant.split is split),
        key=lambda variant: variant.variant_id,
    )
    if not eligible:
        raise BenchmarkError(
            AccessStatus.MODEL_LOAD_FAILURE,
            f"no prepared variants in split {split.value!r} for task {task_name!r}",
        )
    if max_items >= len(eligible):
        return eligible
    picker = random.Random(derive_seed("benchmark_selection", seed, task_name, split.value))
    return sorted(picker.sample(eligible, max_items), key=lambda variant: variant.variant_id)


# ---------------------------------------------------------------------------
# Capture verification
# ---------------------------------------------------------------------------


def verify_capture_point(
    model: LoadedModel, prompt: str, layer: int
) -> tuple[BenchmarkCapture, float]:
    """Check that capture and intervention refer to the same tensor, on real weights.

    Patches `layer` with a known constant vector and reads `layer` back. If the two differ,
    the capture point and the intervention point are not the same place, which is the failure
    that produces results that look fine and mean something else.

    Returns the record and the seconds spent in the capture forward pass.
    """
    validate_layer(layer, model)

    started = time.perf_counter()
    clean = capture_hidden_states(model, prompt, layers=[layer])
    capture_seconds = time.perf_counter() - started

    captured = clean.hidden_states.get(layer)
    if captured is None:
        return (
            BenchmarkCapture(
                layer=layer,
                hook_fired=False,
                shape=[],
                dtype="unknown",
                capture_point_verified=False,
            ),
            capture_seconds,
        )

    sentinel = torch.full((model.hidden_dim,), 0.123, dtype=model.dtype, device=model.device)
    patched = run_with_intervention(
        model,
        prompt,
        InterventionSpec(
            intervention_id="benchmark.capture_check",
            mechanism=Mechanism.ACTIVATION_PATCH,
            mechanism_version="1.0",
            layer=layer,
            position_index=-1,
            strength=1.0,
            source_state_id="benchmark_sentinel",
            analysis_role="harness_validation",
        ),
        InterventionPayload(source_activation=sentinel),
        capture_layers=[layer],
    )
    read_back = patched.hidden_states[layer]
    error = float((read_back.to(torch.float64) - sentinel.to(torch.float64)).abs().max())

    return (
        BenchmarkCapture(
            layer=layer,
            hook_fired=patched.diagnostics is not None,
            shape=list(captured.shape),
            dtype=str(captured.dtype).replace("torch.", ""),
            capture_point_verified=error <= 1e-4,
            max_abs_patch_error=error,
        ),
        capture_seconds,
    )


# ---------------------------------------------------------------------------
# Compute estimate
# ---------------------------------------------------------------------------

# Planned counts, taken from docs/preregistration.md section 9 rather than invented here.
_PLANNED_PROMPTS = 2000
_MODEL_STATES_PER_PROMPT = 2
_CANDIDATES_PER_TRIAL = 4
_FORWARDS_PER_CANDIDATE = 1
_LORA_STEPS = 6000
_LORA_FORWARD_EQUIVALENTS = 3
_PRACTICAL_HOURS = 12.0
_SMALL_VALIDATION_SECONDS = 300.0
_SMALL_VALIDATION_ITEMS = 20


def build_compute_estimate(median_forward_seconds: float) -> ComputeEstimate:
    """Project the cost of the planned experiment from one measured forward time.

    Arithmetic, not a prediction. The assumptions travel with the number so a reader can
    redo it with their own, which matters because the estimate is what a compute decision
    gets made on.
    """
    forward_count = (
        _PLANNED_PROMPTS
        * _MODEL_STATES_PER_PROMPT
        * _CANDIDATES_PER_TRIAL
        * _FORWARDS_PER_CANDIDATE
    )
    cpu_seconds = forward_count * median_forward_seconds
    cpu_hours = cpu_seconds / 3600.0

    lora_hours = (_LORA_STEPS * _LORA_FORWARD_EQUIVALENTS * median_forward_seconds) / 3600.0
    small_validation_seconds = _SMALL_VALIDATION_ITEMS * median_forward_seconds

    small_ok = small_validation_seconds <= _SMALL_VALIDATION_SECONDS
    sweep_ok = cpu_hours <= _PRACTICAL_HOURS
    lora_ok = lora_hours <= _PRACTICAL_HOURS

    return ComputeEstimate(
        assumptions={
            "number_of_prompts": _PLANNED_PROMPTS,
            "model_states_per_prompt": _MODEL_STATES_PER_PROMPT,
            "candidates_per_trial": _CANDIDATES_PER_TRIAL,
            "forward_passes_per_candidate": _FORWARDS_PER_CANDIDATE,
            "median_forward_seconds": median_forward_seconds,
            "planned_counts_source": "docs/preregistration.md section 9",
            "lora_steps": _LORA_STEPS,
            "lora_forward_equivalents_per_step": _LORA_FORWARD_EQUIVALENTS,
            "practical_hours_threshold": _PRACTICAL_HOURS,
            "formula": (
                "estimated_forward_count = number_of_prompts * model_states_per_prompt "
                "* candidates_per_trial * forward_passes_per_candidate; "
                "estimated_cpu_seconds = estimated_forward_count * median_forward_seconds"
            ),
        },
        estimated_forward_count=forward_count,
        estimated_cpu_seconds=cpu_seconds,
        estimated_cpu_hours=cpu_hours,
        lora_estimated_cpu_hours=lora_hours,
        small_clean_validation_practical_on_cpu=small_ok,
        full_sweep_practical_on_cpu=sweep_ok,
        lora_training_practical_on_cpu=lora_ok,
        gpu_rental_recommended=not (sweep_ok and lora_ok),
        notes=[
            "Serial single-prompt forwards, batch size one. Batching would reduce this.",
            "Backward-pass cost is approximated as two extra forwards per step.",
            "Scales linearly from one median forward time and ignores memory pressure.",
            "The estimate informs a decision; it does not make one.",
        ],
    )


# ---------------------------------------------------------------------------
# The benchmark
# ---------------------------------------------------------------------------

_LIMITATIONS = [
    "CPU timing is machine-specific.",
    "This is not a causal self-forecasting experiment.",
    "This run does not test a trained model organism or forecaster.",
    "Accuracy over a handful of items is a scoring smoke check, not a capability measurement.",
    "Latency is measured on one representative prompt at batch size one.",
]


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile.

    Chosen over an interpolating estimator because these samples are tiny (three runs is a
    normal case here) and an interpolated p90 of three points would imply a precision the
    sample does not have.
    """
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def run_benchmark(
    model_config: ModelConfig,
    task_config: TaskConfig,
    model_config_path: str | Path,
    task_config_path: str | Path,
    split: Split,
    max_items: int,
    warmup_runs: int,
    timed_runs: int,
    capture_layer: int | None,
    seed: int,
    output_dir: Path | None,
    offline: bool,
    force: bool,
) -> dict[str, Any]:
    """Load a model, score a few items, measure the cost, and verify the capture path."""
    set_global_seed(seed)

    is_fixture = model_config.kind == "fixture"
    classification = (
        BenchmarkClassification.FIXTURE_SYSTEMS_TEST
        if is_fixture
        else BenchmarkClassification.SYSTEMS_BENCHMARK
    )

    run_id = f"benchmark-{model_config.name}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    directory = Path(output_dir) if output_dir is not None else run_dir(run_id)
    if directory.exists() and any(directory.iterdir()) and not force:
        raise BenchmarkError(
            AccessStatus.MODEL_LOAD_FAILURE,
            f"{directory} already exists and is not empty; pass --force to overwrite it",
        )
    directory.mkdir(parents=True, exist_ok=True)

    try:
        items, variants = load_prepared_task(task_config.name)
    except TaskLoadError as error:
        raise BenchmarkError(AccessStatus.MODEL_LOAD_FAILURE, str(error)) from error
    answer_by_item = {item.item_id: item.answer_label for item in items}

    selected = select_variants(variants, split, max_items, seed, task_config.name)

    access = check_model_access(model_config, offline)
    if access.status not in (
        AccessStatus.CACHED_LOAD,
        AccessStatus.REMOTE_DOWNLOAD,
        AccessStatus.FIXTURE_LOCAL,
    ):
        raise BenchmarkError(access.status, access.detail)
    info("model access check", status=access.status.value, detail=access.detail)

    # Model load. Timed as one number: the centralized loader builds the tokenizer and the
    # model together, and splitting that path just to time it would mean maintaining a second
    # loader.
    load_started = time.perf_counter()
    try:
        with _offline_env(offline):
            model = load_model(model_config)
    except ModelLoadError as error:
        if offline:
            status = AccessStatus.OFFLINE_CACHE_MISS
        elif _is_network_error(error):
            status = AccessStatus.NETWORK_FAILURE
        else:
            status = AccessStatus.MODEL_LOAD_FAILURE
        raise BenchmarkError(status, f"could not load the model: {error}") from error
    model_load_seconds = time.perf_counter() - load_started

    rss_after_load, _, memory_note = read_process_memory()

    layer = capture_layer if capture_layer is not None else model.num_layers // 2
    validate_layer(layer, model)

    # Answer labels must be scoreable before anything is measured. A silent fallback to
    # free-form generation here would change the measured quantity without saying so.
    label_token_ids = resolve_label_token_ids(
        model.tokenizer, task_config.answer_labels, prefix=task_config.label_prefix
    )
    info("resolved answer label tokens", tokens=label_token_ids)

    tokenization_seconds = 0.0
    prompt_tokens: dict[str, int] = {}
    for variant in selected:
        started = time.perf_counter()
        encoded = model.tokenizer(variant.prompt_text, return_tensors="pt")
        tokenization_seconds += time.perf_counter() - started
        prompt_tokens[variant.variant_id] = int(encoded["input_ids"].shape[1])

    representative = selected[0]

    for index in range(warmup_runs):
        capture_hidden_states(model, representative.prompt_text, layers=[])
        info("warmup forward", run=index + 1, of=warmup_runs)

    # Latency on one fixed prompt, kept separate from the evaluation forwards so that the
    # spread reflects run-to-run variation rather than differences in prompt length.
    forward_seconds: list[float] = []
    for index in range(timed_runs):
        started = time.perf_counter()
        capture_hidden_states(model, representative.prompt_text, layers=[])
        elapsed = time.perf_counter() - started
        forward_seconds.append(elapsed)
        info("timed forward", run=index + 1, of=timed_runs, seconds=round(elapsed, 4))

    median_forward = statistics.median(forward_seconds)

    # Evaluation: one forward per item, scored from the four label logits directly.
    evaluation_started = time.perf_counter()
    item_records: list[dict[str, Any]] = []
    correct = 0
    for variant in selected:
        result = capture_hidden_states(model, variant.prompt_text, layers=[])
        scores = score_logits(
            result.next_token_logits, label_token_ids, answer_by_item[variant.item_id]
        )
        correct += int(scores.is_correct)
        item_records.append(
            {
                "variant_id": variant.variant_id,
                "item_id": variant.item_id,
                "framing": variant.framing.value,
                "split": variant.split.value,
                "prompt_tokens": prompt_tokens[variant.variant_id],
                "label_logits": scores.logits,
                "label_probabilities": scores.probabilities,
                "predicted_label": scores.predicted_label,
                "expected_label": scores.correct_label,
                "correct": scores.is_correct,
                "margin": scores.margin,
                "entropy": scores.entropy,
            }
        )
    evaluation_seconds = time.perf_counter() - evaluation_started

    capture_record, capture_forward_seconds = verify_capture_point(
        model, representative.prompt_text, layer
    )
    if not capture_record.capture_point_verified:
        warn(
            "capture point verification failed",
            layer=layer,
            max_abs_patch_error=capture_record.max_abs_patch_error,
        )

    # A single-sample difference. Reported as null when it lands below zero, because at that
    # point the measurement is noise and rounding it up to zero would imply it was resolved.
    overhead = capture_forward_seconds - median_forward
    capture_overhead = overhead if overhead > 0 else None

    _, rss_peak, _ = read_process_memory()
    representative_tokens = prompt_tokens[representative.variant_id]
    total_timed_tokens = representative_tokens * timed_runs
    total_timed_seconds = sum(forward_seconds)

    timing = BenchmarkTiming(
        model_load_seconds=model_load_seconds,
        tokenization_seconds_total=tokenization_seconds,
        warmup_runs=warmup_runs,
        timed_runs=timed_runs,
        forward_seconds_median=median_forward,
        forward_seconds_p90=_percentile(forward_seconds, 0.9),
        forward_seconds_min=min(forward_seconds),
        forward_seconds_max=max(forward_seconds),
        evaluation_seconds_total=evaluation_seconds,
        capture_overhead_seconds=capture_overhead,
        representative_prompt_tokens=representative_tokens,
        total_timed_tokens=total_timed_tokens,
        prefill_tokens_per_second=(
            total_timed_tokens / total_timed_seconds if total_timed_seconds > 0 else None
        ),
    )

    manifest_path = task_manifest_path(task_config.name)
    items_path = private_dir(run_id) if output_dir is None else directory / "private_payloads"
    items_path.mkdir(parents=True, exist_ok=True)
    items_file = write_jsonl(items_path / BENCHMARK_ITEMS, item_records)

    record = BenchmarkRecord(
        run_id=run_id,
        status="completed",
        classification=classification,
        fixture_only=is_fixture,
        model=BenchmarkModelInfo(
            model_id=model.spec.model_id,
            revision=model.spec.revision,
            # The centralized loader pins the tokenizer to the same revision as the weights.
            tokenizer_revision=model.spec.revision,
            device=model.spec.device,
            dtype=model.spec.dtype,
            num_layers=model.num_layers,
            hidden_dim=model.hidden_dim,
            access_status=access.status,
        ),
        task=BenchmarkTaskInfo(
            dataset=f"{task_config.source}:{task_config.source_config or 'default'}",
            split=split,
            item_count=len(selected),
            manifest_hash=hash_file(manifest_path) if manifest_path.exists() else None,
            label_token_ids=label_token_ids,
            label_prefix=task_config.label_prefix,
            scoring_format=SCORING_FORMAT,
        ),
        timing=timing,
        memory=BenchmarkMemory(
            rss_bytes_after_load=rss_after_load,
            rss_bytes_peak=rss_peak,
            measurement=memory_note,
        ),
        evaluation=BenchmarkEvaluation(
            scored_items=len(selected),
            correct_items=correct,
            accuracy=correct / len(selected) if selected else None,
        ),
        capture=capture_record,
        compute_estimate=build_compute_estimate(median_forward),
        limitations=list(_LIMITATIONS),
        config_hashes={
            "model_config": config_hash(model_config_path),
            "task_config": config_hash(task_config_path),
        },
        provenance=[
            ArtifactHashRecord(
                path=str(items_file.name),
                hash=hash_file(items_file),
                size_bytes=items_file.stat().st_size,
                kind="benchmark_items_private",
            )
        ],
    )

    benchmark_file = atomic_write_json(directory / BENCHMARK, record.model_dump(mode="json"))
    environment = environment_snapshot()
    atomic_write_json(directory / ENVIRONMENT, environment)

    run_manifest = RunManifest(
        run_id=run_id,
        phase="systems_benchmark",
        config_path=str(model_config_path),
        config_hash=config_hash(model_config_path),
        seed=seed,
        models=[model.spec],
        counts={"scored_items": len(selected), "timed_runs": timed_runs},
        environment=environment,
        status="complete",
        notes=(
            "Systems benchmark, not a CSF-Bench result. No forecasts, commitments, or "
            "observations exist in this run, so `csf verify run` will not verify it and the "
            "public exporter cannot accept it."
        ),
    )
    atomic_write_json(directory / RUN_MANIFEST, run_manifest.model_dump(mode="json"))

    atomic_write_json(
        directory / ARTIFACT_HASHES,
        [
            ArtifactHashRecord(
                path=path.name,
                hash=hash_file(path),
                size_bytes=path.stat().st_size,
                kind=kind,
            ).model_dump(mode="json")
            for path, kind in (
                (benchmark_file, "benchmark_aggregate"),
                (directory / ENVIRONMENT, "environment"),
                (directory / RUN_MANIFEST, "run_manifest"),
                (items_file, "benchmark_items_private"),
            )
        ],
    )

    info(
        "benchmark complete",
        run_id=run_id,
        classification=classification.value,
        median_forward_seconds=round(median_forward, 4),
        accuracy=record.evaluation.accuracy,
        capture_point_verified=capture_record.capture_point_verified,
    )
    payload = record.model_dump(mode="json")
    payload["output_dir"] = str(directory)
    return payload


__all__ = [
    "AccessCheck",
    "BenchmarkError",
    "build_compute_estimate",
    "check_model_access",
    "read_process_memory",
    "run_benchmark",
    "select_variants",
    "verify_capture_point",
]
