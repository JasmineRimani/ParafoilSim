# PIMENTO

`PIMENTO` is a lightweight Python toolkit for preliminary parafoil sizing and low-fidelity trajectory simulation. It is designed for early-stage engineering studies, rapid trade analysis, and research prototyping.

It is inspired by the public paper "An Approach to the Preliminary Sizing and Performance Assessment of Spaceplanes' Landing Parafoils" and by the author's earlier research codebase.

What is included:

- preliminary parafoil sizing from payload mass, trim speed, and target lift-to-drag ratio
- a generic aspect-ratio-driven aerodynamic surrogate for early trades
- a 3D point-mass guidance model with homing, loiter, energy management, and final approach phases
- an experimental nine-DOF starter model for research exploration (still to be validated)

## Install

```bash
python -m pip install -e .
```

## Quick Start

```bash
PYTHONPATH=src python examples/public_case.py
```

The example uses a paper-like public case with:
- payload mass near `2028 kg`
- nominal parafoil `L/D` near `4`
- a preliminary design sweep around a `280 m^2` class canopy

## Notes on Scope

PIMENTO is intentionally low fidelity and focuses on early-stage design insight, not high-accuracy prediction.
The aerodynamic model is simplified and not tuned to specific systems
Results should be interpreted as trend-level guidance, not final design data
The nine_dof module is experimental and not validated against flight data

## Sources

This work is inspired by:

[An Approach to the Preliminary Sizing and Performance Assessment of Spaceplanes' Landing Parafoils](https://www.mdpi.com/2226-4310/9/12/823)

## Intended Use

PIMENTO is best suited for:

Early design exploration
Trade studies
Academic research
Concept validation

