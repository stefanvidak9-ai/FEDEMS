# Step 31: Daily 1D-CNN Architecture

## Purpose and Scope

`DailyHeatingCNN` is a compact PyTorch model for one-day-ahead regression of
residential space-heating energy. It maps seven preceding days of features to
one scalar: the next day's total `HtSpace` in kWh/day. The architecture is
implemented independently of data loading, normalization, training, and
federated orchestration.

The design follows the general federated CNN forecasting principle discussed
by Moradzadeh et al. (2022), *A Secure Federated Deep Learning-Based Approach
for Heating Load Demand Forecasting in Building Environment*, IEEE Access. It
is an FEDEMS-specific lightweight starting architecture and is **not** claimed
to reproduce that paper's model or experiments.

## Input and Output

The model expects float32 input with shape
`[batch_size, num_features, sequence_length]`. The feature dimension is
configurable and corresponds to Conv1D channels. FEDEMS initial experiments
use seven chronological days, so an illustrative batch may have shape
`[32, 10, 7]`; `10` is only an example and is not a finalized dataset feature
count. Feature names, NPZ keys, and normalization are intentionally left to
the data-interface agreement with Person A.

The returned tensor has shape `[batch_size, 1]`. Each row is an unconstrained
linear prediction for the following day's `HtSpace`; no activation is applied
to the regression output. The architecture accepts batch size one and uses
adaptive pooling, so it can also process other positive sequence lengths.

## Architecture

| Stage | Operation | Output for `[B, 10, 7]` |
| --- | --- | --- |
| Input | Float32 feature channels | `[B, 10, 7]` |
| Temporal feature extractor 1 | Conv1D, 16 channels, kernel 3, padding 1; ReLU | `[B, 16, 7]` |
| Temporal feature extractor 2 | Conv1D, 32 channels, kernel 3, padding 1; ReLU | `[B, 32, 7]` |
| Pool | Adaptive average pool to length 1 | `[B, 32, 1]` |
| Regression head | Flatten pooled features; Linear(32, 1) | `[B, 1]` |

The convolutions share weights over the short temporal axis and can learn
local patterns across adjacent days. Global average pooling keeps the head
small and independent of sequence length. The two-layer design is a
conservative starting point given the limited number of daily observations
available per building; it does not eliminate overfitting risk.

The default trainable parameter count depends on `num_features` $F$:

$$
P(F) = (3F\cdot16 + 16) + (3\cdot16\cdot32 + 32) + (32\cdot1 + 1)
     = 48F + 1617.
$$

For the illustrative `F = 10`, this is **2,097 trainable parameters**, or
**8,388 bytes (about 8.19 KiB)** when stored as float32. This is a parameter
storage estimate, not a measured runtime memory or speed benchmark.

## Initialization and Forward Pass

The constructor accepts `num_features` and optional convolution widths and
kernel size. Defaults are 16 and 32 channels with a three-day kernel. The
kernel must be a positive odd integer so same-length padding is unambiguous.
No configuration-file convention exists for model settings yet, so the
constructor is the current interface; a future experiment config can pass
these values without changing the model API.

```python
import torch

from src.models import DailyHeatingCNN

model = DailyHeatingCNN(num_features=10).cpu().eval()
synthetic_inputs = torch.randn(32, 10, 7, dtype=torch.float32)
with torch.no_grad():
    predictions = model(synthetic_inputs)
print(predictions.shape)  # torch.Size([32, 1])
```

This example uses synthetic tensors only to verify model functionality. Its
outputs are not heating forecasts or research results. Actual inputs must
follow the feature order and preprocessing contract agreed with Person A.

## Tests and Runtime

Install dependencies and run the CNN tests and full suite:

```powershell
python -m pip install -r requirements.txt
python -m pytest tests/test_cnn.py
python -m pytest
```

The implementation uses ordinary `torch.nn` CPU operations and does not
require CUDA. Its `state_dict` is independent of training orchestration, so
the same architecture and parameter layout can be instantiated for
centralized training, independent local models, and future Flower/FedAvg
clients. This step does not implement any of those training workflows.

## Limitations

Daily observations per building may be few relative to model capacity, and
buildings can have different lengths and missing periods. Use the existing
Step 30 baselines, carefully separated train/validation/test dates, and
consistent client/date samples when assessing later models. Feature selection,
normalization, missing-data policy, and dataset adapters remain unconfirmed;
this architecture makes no assumptions about them. Raspberry Pi runtime,
memory, and training performance must be measured in Phase 7 rather than
inferred from parameter count.