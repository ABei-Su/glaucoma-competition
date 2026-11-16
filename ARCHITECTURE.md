# Architecture

`data.py` owns labeled split boundaries and deterministic per-image preprocessing. `model.py` owns the ImageNet-compatible backbone. `train.py` owns supervised training, validation, selection and atomic checkpoints. `predict.py` owns frozen image inference and CSV production. `cost.py` imports the same Predictor and includes file input/output in timing. `utils.py` validates scores and CSVs and centralizes training lifecycle transitions.

Raw → derived lossless train/valid cache → training/validation → frozen checkpoint → test predictions. No reverse path from test predictions to training. External network access is confined to fixed public resource URLs in fetch_pretrain.py. Prediction never downloads code or weights. Checkpoints must be locally generated or trusted; inference uses weights_only=True. Resume checkpoints contain Python/NumPy RNG state and use trusted local loading.

The run lock prevents concurrent writers. Completed checkpoints are atomically replaced. On a caught failure status is written and lock removed; hard termination may leave a lock which requires verifying its PID before removal. Epoch replay is bounded to the current incomplete epoch. No automatic infinite training retry.
