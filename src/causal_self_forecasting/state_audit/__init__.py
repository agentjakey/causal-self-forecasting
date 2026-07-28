"""The BlueDot state-dependence execution path.

Everything in this package is specific to the state-dependence arm: its candidate shapes, its
target, its run manifest, and its verifier. The benchmark's trial generation, four-candidate
builder, and `delta_margin` observations are untouched and keep their own meaning.

Nothing here is a scientific result. An engineering smoke run proves the plumbing works on real
weights; it selects nothing and measures nothing about the model's abilities.
"""

from __future__ import annotations
