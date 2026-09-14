# CDR_v1

Model code for compensation and intervention portfolio optimisation in
carbon dioxide removal.

## Approach

Every object in the framework is a sum of polynomial exponential terms, a
class closed under convolution and integration. All results therefore follow
in closed form, with no time grid and no quadrature anywhere.

## Contents

The `cipo` package holds the model:

- `expsum.py` the polynomial exponential algebra that everything rests on
- `climate.py` impulse responses, climate metrics, carbon feedback
- `profiles.py` intervention families, compared through mean storage time
- `pathways.py` multi species emission pathways
- `portfolio.py` Gram matrix, neutrality constraint, optimisers
- `threshold.py` horizon sensitivity and optimal durability
- `uncertainty.py` Monte Carlo, robust design, Sobol indices
- `dynamic.py` deployment timing as a convex quadratic programme

The `scripts` folder regenerates every table and figure. The `tests` folder
covers the package.

## Usage

Python 3 with the packages listed in `requirements.txt`.

```
pip install -r requirements.txt
python3 scripts/run_all.py     # regenerate all tables and figures
python3 -m pytest tests -q     # run the test suite
```

Outputs are written to `results` and `figures`, which the scripts create.

## Licence

See `LICENSE`.
