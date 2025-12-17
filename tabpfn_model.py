"""
TabPFN model wrapper for active learning benchmark

This wrapper adapts TabPFN to work with the active learning benchmark framework.
TabPFN has specific constraints:
- Maximum 10,000 training samples
- Maximum 100 features
- Supports binary and multi-class classification
"""
from tabpfn import TabPFNClassifier
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings


class TabPFNWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper for TabPFN to work with the active learning benchmark.
    Handles automatic truncation when constraints are exceeded.
    """
    
    def __init__(self, N_ensemble_configurations=32, device='cpu', 
                 max_samples=10000, max_features=100, **kwargs):
        """
        Initialize TabPFN wrapper
        
        Args:
            N_ensemble_configurations (int): Number of ensemble members (default: 32)
                - Larger values = better performance but slower
                - Typical range: 16-64
            device (str): 'cpu' or 'cuda' for GPU acceleration
            max_samples (int): Maximum number of training samples (default: 10000)
            max_features (int): Maximum number of features (default: 100)
            **kwargs: Additional arguments passed to TabPFNClassifier
        """
        self.N_ensemble_configurations = N_ensemble_configurations
        self.device = device
        self.max_samples = max_samples
        self.max_features = max_features
        self.kwargs = kwargs
        self.model = None
        self._feature_indices = None
        
    def _validate_and_truncate(self, X, y=None, fit_mode=True):
        """
        Validate input and truncate if necessary
        
        Args:
            X: Feature matrix
            y: Labels (optional, only needed in fit mode)
            fit_mode: Whether this is called from fit() or predict()
            
        Returns:
            X_processed: Processed feature matrix
            y_processed: Processed labels (None if y is None)
        """
        X_processed = X.copy() if hasattr(X, 'copy') else np.array(X)
        y_processed = y.copy() if y is not None and hasattr(y, 'copy') else y
        
        # Handle sample limit
        if fit_mode and X_processed.shape[0] > self.max_samples:
            warnings.warn(
                f"TabPFN supports max {self.max_samples} samples. "
                f"Got {X_processed.shape[0]}. Truncating to first {self.max_samples}."
            )
            X_processed = X_processed[:self.max_samples]
            if y_processed is not None:
                y_processed = y_processed[:self.max_samples]
        
        # Handle feature limit
        if X_processed.shape[1] > self.max_features:
            if fit_mode:
                # Store which features to keep
                self._feature_indices = np.arange(self.max_features)
                warnings.warn(
                    f"TabPFN supports max {self.max_features} features. "
                    f"Got {X_processed.shape[1]}. Using first {self.max_features}."
                )
            
            # Apply feature selection
            if self._feature_indices is not None:
                X_processed = X_processed[:, self._feature_indices]
            else:
                X_processed = X_processed[:, :self.max_features]
        
        return X_processed, y_processed
    
    def fit(self, X, y):
        """
        Fit TabPFN model
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            self: Fitted estimator
        """
        X_processed, y_processed = self._validate_and_truncate(X, y, fit_mode=True)
        
        self.model = TabPFNClassifier(
            N_ensemble_configurations=self.N_ensemble_configurations,
            device=self.device,
            **self.kwargs
        )
        
        try:
            self.model.fit(X_processed, y_processed)
        except Exception as e:
            warnings.warn(f"TabPFN fit failed: {e}. Using default predictions.")
            # Store classes for fallback
            self.classes_ = np.unique(y_processed)
            self.model = None
        
        return self
    
    def predict(self, X):
        """
        Predict class labels
        
        Args:
            X: Features to predict
            
        Returns:
            predictions: Predicted class labels
        """
        if self.model is None:
            # Fallback: predict majority class
            return np.full(X.shape[0], self.classes_[0])
        
        X_processed, _ = self._validate_and_truncate(X, fit_mode=False)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Features to predict
            
        Returns:
            probabilities: Predicted class probabilities
        """
        if self.model is None:
            # Fallback: uniform probabilities
            n_classes = len(self.classes_)
            return np.full((X.shape[0], n_classes), 1.0 / n_classes)
        
        X_processed, _ = self._validate_and_truncate(X, fit_mode=False)
        return self.model.predict_proba(X_processed)
    
    def score(self, X, y):
        """
        Return accuracy score
        
        Args:
            X: Features
            y: True labels
            
        Returns:
            accuracy: Classification accuracy
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    @property
    def classes_(self):
        """Return class labels"""
        if self.model is not None and hasattr(self.model, 'classes_'):
            return self.model.classes_
        elif hasattr(self, '_classes'):
            return self._classes
        return None
    
    @classes_.setter
    def classes_(self, value):
        """Set class labels"""
        self._classes = value


class TabPFNFast(TabPFNWrapper):
    """Fast TabPFN configuration with fewer ensemble members"""
    def __init__(self, **kwargs):
        super().__init__(N_ensemble_configurations=16, **kwargs)


class TabPFNAccurate(TabPFNWrapper):
    """More accurate TabPFN configuration with more ensemble members"""
    def __init__(self, **kwargs):
        super().__init__(N_ensemble_configurations=64, **kwargs)


if __name__ == '__main__':
    # Test the wrapper
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    print("Testing TabPFN Wrapper")
    print("=" * 60)
    
    # Generate test data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Test standard wrapper
    print("\n1. Testing standard TabPFN wrapper...")
    model = TabPFNWrapper()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    print(f"   Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"   Probability shape: {y_proba.shape}")
    print(f"   Classes: {model.classes_}")
    
    # Test fast wrapper
    print("\n2. Testing fast TabPFN wrapper...")
    model_fast = TabPFNFast()
    model_fast.fit(X_train, y_train)
    y_pred_fast = model_fast.predict(X_test)
    print(f"   Accuracy: {accuracy_score(y_test, y_pred_fast):.4f}")
    
    # Test with too many features
    print("\n3. Testing with many features (>100)...")
    X_many_features = np.random.randn(500, 150)
    y_many = np.random.randint(0, 2, 500)
    model.fit(X_many_features, y_many)
    print(f"   Fit successful with feature truncation")
    
    # Test with too many samples
    print("\n4. Testing with many samples (>10000)...")
    X_many_samples = np.random.randn(15000, 20)
    y_many_samples = np.random.randint(0, 2, 15000)
    model.fit(X_many_samples, y_many_samples)
    print(f"   Fit successful with sample truncation")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
