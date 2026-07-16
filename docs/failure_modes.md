# Failure-mode taxonomy

Fifteen ways a forecasting result could look good and mean nothing. Each has a detection
method that is part of the design rather than something to be attempted later if a reviewer
asks.

Status column: `not tested` everywhere, because no experiment has been run.

| # | Failure mode | What it looks like | How it is detected | Status |
| --- | --- | --- | --- | --- |
| 1 | Surface shortcut | Forecasts track prompt wording, not internal state | Prompt-only TF-IDF baseline matches the state-conditioned model | not tested |
| 2 | State-insensitive forecasting | The state input is ignored | Shuffled-state and wrong-example-state conditions do not hurt | not tested |
| 3 | Intervention-label shortcut | The candidate's label gives the answer away | Opaque ids; the public view of a random control is identical to a real steer | not tested |
| 4 | Linear-only forecasting | Works only for tiny perturbations | Strength-extrapolation split; comparison against the gradient baseline | not tested |
| 5 | Mechanism overfitting | Learns one operation, not the causal structure | Held-out activation patching (H4) | not tested |
| 6 | Layer overfitting | Works only at trained layers | Layer split: unseen earlier, later, and middle layers | not tested |
| 7 | Magnitude miscalibration | Sign right, size wrong | Signed error and interval coverage reported separately from sign accuracy | not tested |
| 8 | Evaluation-framing sensitivity | Reports change with framing while behavior does not | Evaluation, deployment, and neutral wrappers with paraphrases (H6) | not tested |
| 9 | Clean-versus-adapted state confusion | Cannot tell which model produced a state | Same-prompt state swap; false-positive rate on clean models | not tested |
| 10 | Invalid or evasive report formatting | The model does not emit scorable output | Invalid-JSON and repair rates reported; never silently repaired | not tested |
| 11 | Candidate-order dependence | Position in the list drives the forecast | Order randomized per trial; order sensitivity measured | not tested |
| 12 | Detecting disturbance without predicting effect | Knows something happened, not what | Random control produces a disturbance with a different effect; compare | not tested |
| 13 | Predicting effect without identifying the concept | Gets the margin right, the mechanism wrong | Bias-suppression prediction scored separately from delta margin | not tested |
| 14 | Probe and verbal-report dissociation | The probe reads it out, the model cannot say it | Linear probe and verbal reporter compared on identical trials | not tested |
| 15 | Confidence collapse under distribution shift | Calibration dies off-distribution | ECE reported per split, including every held-out condition | not tested |

## Notes on the ones that are easiest to get wrong

**#3, intervention-label shortcut.** This is subtler than removing names. If the public
description of a candidate said `random_add`, a forecaster could predict "no effect" for every
control without looking at the state at all, and the random-direction control would silently
stop being a control. The public view therefore describes a matched random steer and a real
bias steer identically: same operation, same layer, same strength, different opaque id. Pinned
by `test_public_view_makes_the_random_control_indistinguishable`.

**#12, detecting disturbance without predicting effect.** A forecaster can learn "a large
perturbation was applied, so something will change" without knowing what. The norm-matched
random control is what separates these: it is the same size of disturbance with a different
meaning. If effects are predicted equally well for the real direction and the control, the
method has learned disturbance magnitude, not causal structure.

**#2 versus #9.** These are different. #2 is not using the state at all. #9 is using the state
but being unable to distinguish the clean model from the adapted one under the same prompt.
A method can fail #9 while passing #2, which is why the shuffled-state and wrong-model-state
conditions are both run.

## Reporting

The dashboard's failure-mode page shows evidence for and against each mode, with sample
counts. Modes with evidence against the project's own hypotheses stay visible. A failure-mode
page that only ever reports clean passes is not evidence of a good method; it is evidence of a
weak test.
