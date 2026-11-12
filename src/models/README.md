# Models

The reusable Step 31 daily heating 1D-CNN is implemented in
[`cnn.py`](cnn.py) as `DailyHeatingCNN`. It defines only the architecture and
tensor shape contract. Steps 32–34 train and evaluate it through
[`src/training`](../training/README.md); federated orchestration remains future
work in Phase 8.