# PIMENTO

`PIMENTO` is a Python starter for preliminary parafoil sizing and low-fidelity trajectory simulation.

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

The aerodynamic surrogate is intentionally low fidelity. The paper itself notes that aerodynamic modelling remains the biggest open point for generalization, so this package keeps the model simple and transparent instead of embedding tuned partner-specific curves.

The `nine_dof` module should be read as a public research starter: it gives a generic coupled canopy-payload structure with joint stiffness and damping, but it is not a validated recreation of the original case-study model.

## Sources

- Paper: https://www.mdpi.com/2226-4310/9/12/823

## License

