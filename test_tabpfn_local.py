
import torch
from tabpfn import TabPFNClassifier
import numpy as np

# Try to load with local checkpoint
ckpt_path = "/home/hosh/.cache/tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt"

print(f"Loading TabPFN with model_path={ckpt_path}")
try:
    classifier = TabPFNClassifier(model_path=ckpt_path, n_estimators=8)
    print("Success!")
    
    # Test fit
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    classifier.fit(X, y)
    print("Fit success!")
    
    # Test predict
    y_pred = classifier.predict(X[:5])
    print(f"Predict success: {y_pred}")
    
except Exception as e:
    print(f"Failed: {e}")
    import traceback
    traceback.print_exc()
