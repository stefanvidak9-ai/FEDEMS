# Evaluation

Reserved for shared forecasting metrics and monitoring utilities. Step 30's
baseline-specific CSV evaluation currently remains in `src/baselines/evaluate.py`;
move or generalize shared code only when a later model needs it and tests can
preserve the existing baseline interface.