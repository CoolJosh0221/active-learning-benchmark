import time
import numpy as np
from src.models.tabpfn_model import TabPFNWrapper
import torch


def test_large_inference():
    print("Testing Large Inference with TabPFN")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Generate large dataset
    N_TRAIN = 100
    N_TEST = 20000
    N_FEATURES = 50

    print(f"Generating data: {N_TRAIN} train, {N_TEST} test samples...")
    X_train = np.random.randn(N_TRAIN, N_FEATURES)
    y_train = np.random.randint(0, 2, N_TRAIN)

    X_test = np.random.randn(N_TEST, N_FEATURES)

    # Initialize model
    model = TabPFNWrapper(device="cuda", N_ensemble_configurations=32)

    # Fit
    print("Fitting model...")
    model.fit(X_train, y_train)

    # Predict
    print(f"Predicting on {N_TEST} samples...")
    start_time = time.time()
    try:
        probs = model.predict_proba(X_test)
        duration = time.time() - start_time
        print(f"Success! Prediction took {duration:.2f} seconds.")
        print(f"Output shape: {probs.shape}")
    except Exception as e:
        print(f"Failed! Error: {e}")


if __name__ == "__main__":
    test_large_inference()
