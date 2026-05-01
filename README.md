# bmi-dummy

A no-op [Basic Model Interface (BMI)](https://bmi.readthedocs.io/) model for use with [NextGen / NGIAB](https://github.com/CIROH-UA). It accepts any input variables and always returns `0.0` for every output at every time step.

Authored by Claude.

## Install

```bash
pip install -e .
```

## Usage

```python
import numpy as np
from bmi_dummy import BmiDummy

model = BmiDummy()
model.initialize("/dev/null")   # or a YAML config path, or ""

while model.get_current_time() < model.get_end_time():
    model.set_value("input", np.array([42.0]))
    model.update()
    dest = np.zeros(1)
    model.get_value("output", dest)   # always 0.0

model.finalize()
```

## Optional YAML config

```yaml
start_time: 0.0
end_time: 86400.0
time_step: 3600.0
input_vars: [rainfall, temperature]
output_vars: [streamflow]
```

All keys are optional — `/dev/null` (empty file) falls back to defaults.
