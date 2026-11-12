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
configurable and corresponds to Conv1D channels. Phase 4/5 defines the
production FEDEMS contract as four features over seven preceding days, so the
usual batch shape is `[32, 4, 7]`. The feature order is
`HtSpace_kWh_day`, `Tout_mean_C`, `SolGlob_kWh_m2_day`, and
`WindSpd_mean_m_s`; Phase 5 supplies the chronological partitions and
training-only global z-score normalization. Steps 32–34 consume those ready
NPZ windows without rebuilding preprocessing.

The returned tensor has shape `[batch_size, 1]`. Each row is an unconstrained
linear prediction for the following day's `HtSpace`; no activation is applied
to the regression output. The architecture accepts batch size one and uses
adaptive pooling, so it can also process other positive sequence lengths.

## Architecture

| Stage | Operation | Output for `[B, 4, 7]` |
| --- | --- | --- |
| Input | Float32 feature channels | `[B, 4, 7]` |
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

For the FEDEMS `F = 4`, this is **1,809 trainable parameters**, or
**7,236 bytes (about 7.07 KiB)** when stored as float32. This is a parameter
storage estimate, not a measured runtime memory or speed benchmark.

## Initialization and Forward Pass

The constructor accepts `num_features` and optional convolution widths and
kernel size. Defaults are 16 and 32 channels with a three-day kernel. The
kernel must be a positive odd integer so same-length padding is unambiguous.
The Step 32–34 runner takes the feature count from the Phase 5 scaler metadata
and records the model configuration with each result.

```python
import torch

from src.models import DailyHeatingCNN

model = DailyHeatingCNN(num_features=4).cpu().eval()
synthetic_inputs = torch.randn(32, 4, 7, dtype=torch.float32)
with torch.no_grad():
    predictions = model(synthetic_inputs)
print(predictions.shape)  # torch.Size([32, 1])
```

This example uses synthetic tensors only to verify model functionality. Its
outputs are not heating forecasts or research results. Experiment inputs
follow the finalized Phase 5 feature order and preprocessing contract.

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
buildings can have different characteristics. Steps 32–34 use the existing
Step 30 baselines, finalized Phase 5 train/validation/test dates, and identical
client/date samples for comparison. Raspberry Pi runtime, memory, and training
performance must be measured in Phase 7 rather than inferred from parameter
count.