# InterSelf

Anonymous repository for paper submission. This repository contains the core
implementation of InterSelf-DreamerV3 for learning world models when action
execution is hidden and history dependent.

InterSelf distinguishes the action selected by the policy from the action
representation used by the world model. A recurrent execution state predicts a
categorical action channel and a persistence coefficient. Their output is an
effective action that conditions both posterior world-model learning and latent
imagination.

The code is provided as a small overlay for the public
[DreamerV3](https://github.com/danijar/dreamerv3) JAX/Ninjax implementation.
The upstream encoder, decoder, actor--critic, replay system, and training loop
remain unchanged.

## Repository structure

```text
interself/
  rssm_interself.py       core InterSelf RSSM implementation
  selfatari.py            hidden action-execution Atari wrapper

integration/
  install.py              installs and registers the overlay in DreamerV3

configs/
  interself_atari.yaml    method and environment configuration

evaluation/
  summarize_scores.py     fixed-mode result aggregation

tests/
  test_channel_math.py    action-channel invariants
  test_evaluation.py      aggregation invariants
  test_installer.py       DreamerV3 integration checks
```

## Core mechanism

Let **aₜ** denote the intended one-hot action. InterSelf predicts a
row-stochastic categorical channel **Cₜ** and a persistence value **ρₜ**. The
complete action transformation is:

| Stage | Update | Meaning |
|---|---|---|
| Categorical channel | **Cₜ = diag(1 − αₜ) + diag(αₜ)Dₜ** | Blend the identity mapping with a learned destination distribution. |
| Action mapping | **āₜ = aₜ Cₜ** | Map the intended action through the learned channel. |
| Temporal persistence | **ãₜ = (1 − ρₜ)āₜ + ρₜãₜ₋₁** | Mix the mapped action with the previous effective action. |
| World transition | **hₜ₊₁ = RSSMCore(hₜ, zₜ, ãₜ)** | Condition the RSSM on the inferred effective action. |

Each row of **Dₜ** is produced by a softmax. Consequently, **Cₜ**, **āₜ**, and
**ãₜ** remain valid categorical distributions. The channel is initialized near
the identity mapping and changes only when the observed transitions support a
deviation.

The same operator is used in `observe()` and `imagine()`. This keeps posterior
learning and policy imagination under the same learned action semantics.

### Recurrent execution state

`sdeter` stores the recurrent execution state. It is updated from the previous
execution state, the RSSM stochastic state, the intended action, and the
previous and current effective actions. This allows the action channel to
represent persistent or history-dependent execution changes instead of only
independent action noise.

### Auxiliary objectives

The implementation adds two losses to the standard DreamerV3 model objective:

- `selfeff` predicts the change in encoded observation tokens from the inferred
  effective action;
- `selfreg` keeps the categorical channel and persistence close to identity
  unless transition evidence supports a deviation.

Their released values and the execution-state dimensions are recorded in
`configs/interself_atari.yaml`.

## Capacity-matched control

`InterSelf-Identity` is included in the same implementation through
`identity_action=True`. It keeps the recurrent execution state, categorical
channel, persistence head, effect predictor, regularizers, auxiliary losses,
parameters, and gradient paths. Only the action supplied to the RSSM transition
is changed:

| Variant | Action supplied to the RSSM |
|---|---|
| **InterSelf** | inferred effective action **ãₜ** |
| **InterSelf-Identity** | intended action **aₜ** |

This comparison isolates whether the learned execution representation should
mediate the world transition. It is not a comparison between different model
capacities.

## Hidden action execution

`SelfInterventionAtari` implements the following execution modes:

| Mode | Executed action |
|---|---|
| `normal` | intended action |
| `sticky` | previous executed action with probability **ρ** |
| `lazy` | `NOOP` with probability **ρ** |
| `confuse_lr` | LEFT/RIGHT semantics exchanged with probability **ρ** |
| `fire_dropout` | FIRE-containing action replaced by `NOOP` with probability **ρ** |

The mode and executed-action diagnostics are emitted only under `log/` keys.
DreamerV3 filters these fields before constructing the agent observation, so
the model receives neither the corruption identity nor the executed-action
label.

## Installation

Install DreamerV3 and its normal dependencies first, then run:

```bash
python -m pip install -r requirements-overlay.txt
python integration/install.py --repo /path/to/dreamerv3
```

The installer copies the two InterSelf modules and registers:

- `InterSelfRSSM` in the DreamerV3 dynamics registry;
- `SelfInterventionAtari` in the environment registry;
- InterSelf and InterSelf-Identity configuration entries.

It creates `*.bak_interself` files before editing upstream files and stops if
the expected DreamerV3 registry blocks are not present.

## Running the three model variants

InterSelf:

```bash
python dreamerv3/main.py \
  --configs atari size12m interself interself_size12m \
  --task selfatari_<game>
```

Capacity-matched InterSelf-Identity:

```bash
python dreamerv3/main.py \
  --configs atari size12m interself interself_size12m interself_identity \
  --task selfatari_<game>
```

DreamerV3 through the same hidden-execution environment:

```bash
python dreamerv3/main.py \
  --configs atari size12m \
  --task selfatari_<game>
```

Use the same upstream model-size preset and environment configuration for all
methods in a comparison.

## Fixed-mode evaluation

For evaluation, hold one execution mode fixed for the complete episode and
disable native ALE sticky actions:

```text
--env.selfatari.modes <mode>
--env.selfatari.switch never
--env.selfatari.switch_every 0
--env.selfatari.sticky False
```

Score files can be organized as:

```text
results/<method>/<game>/<mode>/<run_name>/scores.jsonl
```

Then run:

```bash
python evaluation/summarize_scores.py \
  --root /path/to/results \
  --output /path/to/summary
```

The script first averages episodes within each independent run, then averages
run means within each mode. The corrupted score is the unweighted mean over the
available non-normal modes for the same game. Raw returns are never averaged
across different games.

## Tests

```bash
python -m unittest discover -s tests -v
```

The dependency-light tests cover row-stochasticity, the identity limit,
temporal persistence, the capacity-control branch, fixed-mode aggregation, and
the DreamerV3 registry edits.

## Release scope

The repository contains the method-specific source needed to inspect and run
InterSelf. Machine-specific scheduler files, private paths, checkpoints, raw
logs, experiment-management records, and tuning history are intentionally not
included.

## License

This overlay follows the MIT license used by DreamerV3. See `LICENSE` for the
license text and upstream attribution.
