# utils folder

This folder contains the core Python modules used by the VEGF-GLUT1 notebooks.

## Contents

- `model_params.py`  
  Shared parameter dataclasses and helper functions.

- `fixed_neighborhood.py`  
  Core equations and integrators for the fixed-neighborhood model.

- `evolving_neighborhood.py`  
  Core equations and integrators for the evolving-neighborhood model.

- `models_therapy.py`  
  Therapy-enabled evolving-neighborhood model, including positivity-preserving integration.

- `__init__.py`  
  Marks `utils/` as an importable package.

## Suggested notebook imports

```python
from utils.model_params import fixedN_defaults, evolving_defaults
from utils.fixed_neighborhood import integrate_fixedN, integrate_teamopt
from utils.evolving_neighborhood import integrate_ess, integrate_teamopt
from utils.models_therapy import integrate_ess as integrate_ess_therapy
```

## Why this structure?

This organization keeps the notebooks lighter and makes the project easier to read on GitHub. It also avoids duplicating parameter classes across multiple files.

## Important cleanup already made

The original `evolving_neighborhood.py` both imported `EvolvingParams` from `model_params.py` and redefined a second `EvolvingParams` class locally. That duplication has been removed in this cleaned version so the project uses one shared parameter definition.
