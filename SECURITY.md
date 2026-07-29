# Security and safety

## Research safety boundaries

The design includes a model organism with a planted behavioral bias. **It has not been built and
is deferred**, and the active BlueDot state-dependence arm uses no adapter at all; see
`docs/deferred_work.md`. The boundaries below are fixed in advance and are not negotiable if that
work resumes.

**The organism would be benign by construction.** It is a LoRA adapter that prefers one answer
position on four-choice questions under a deployment-like wrapper, while preserving accuracy
under an evaluation-like wrapper. The harm ceiling is answering a multiple-choice question
with the wrong letter.

It is described as a **benign, evaluation-conditioned task-bias model organism**. It is not
scheming, not deceptive, and does not have goals. Calling it those things would overclaim what
was built and misrepresent what the results mean.

**This project does not build:**

* harmful goals or capabilities;
* safety-filter evasion;
* autonomous behavior or agentic loops;
* tool use or credential access;
* real-world deception;
* persistent or malicious backdoors;
* anything that transfers to a deployed system.

The organism would exist for one reason: a forecaster needs a known internal thing to be right or
wrong about. If a less capable organism would serve, it should be used instead. The active arm
avoids the question entirely by measuring the effect of a constructed intervention on the model's
own clean preferred answer, which needs no planted behavior.

## Secrets and private artifacts

Never commit:

* commitment salts (`*_salt.txt`, `commitment_salts/`);
* selection seeds (`selection_seed.txt`, `.secrets/`);
* private intervention payloads (`private_payloads/`);
* API tokens or Hugging Face credentials;
* model weights, adapters, or activation caches.

`.gitignore` covers these. If a salt or seed is committed, the affected run's commitments no
longer demonstrate anything and the run must be regenerated rather than patched.

## Model weights

Gemma 3 weights are gated and licensed. This repository pins revisions and never redistributes
weights. Users obtain them from Hugging Face under Google's license.

## Reporting a vulnerability

For a security issue in this code, open a GitHub issue for anything non-sensitive. For
something sensitive, contact the maintainer directly rather than filing publicly.

This is a research repository. It runs untrusted model weights only if you point it at them,
and `trust_remote_code` defaults to false in every config.
