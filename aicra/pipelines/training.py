"""Training pipeline for AICRA models."""

from __future__ import annotations

import mlflow
import mlflow.sklearn
import numpy as np
from typing import Literal

from ..config import Settings
from ..core.data import Dataset
from ..models.lightgbm import train_bagged_lightgbm
from .features_pe import build_pe_features


class TrainingPipeline:
    """Training pipeline with MLflow logging and reproducibility."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def run(
        self,
        train_data: Dataset,
        model_type: Literal["lgbm", "ffnn"] = "lgbm",
        model_name: str = "bagged_lightgbm",
        experiment_name: str | None = None,
        seeds: int = 5,
    ) -> str:
        """Train model and log to MLflow."""
        
        # Set up MLflow
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Log git commit and parameters for reproducibility
            mlflow.log_param("git_commit", self.settings.git_commit)
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("seeds", seeds)
            
            # Log training parameters
            mlflow.log_params({
                "learning_rate": self.settings.learning_rate,
                "num_leaves": self.settings.num_leaves,
                "n_estimators": self.settings.n_estimators,
                "subsample": self.settings.subsample,
                "colsample_bytree": self.settings.colsample_bytree,
                "goss": self.settings.goss,
                "class_weight": self.settings.class_weight,
            })

            # Extract PE features if needed
            if hasattr(train_data, 'file_paths') and train_data.file_paths is not None:
                mlflow.log_param("feature_type", "pe_static")
                pe_features = build_pe_features(train_data.file_paths)
                # Combine with existing features
                X = np.hstack([train_data.features.values, pe_features.values])
            else:
                mlflow.log_param("feature_type", "ember")
                X = train_data.features.values

            # Train model based on type
            if model_type == "lgbm":
                model = self._train_lightgbm(X, train_data.labels.values, seeds)
            elif model_type == "ffnn":
                model = self._train_ffnn(X, train_data.labels.values, seeds)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Save model locally
            model_path = self.settings.models_dir / f"{model_name}.joblib"
            model.save(model_path)

            # Log model path
            mlflow.log_artifact(str(model_path))

            return str(model_path)
    
    def _train_lightgbm(self, X: np.ndarray, y: np.ndarray, seeds: int) -> Any:
        """Train LightGBM model with histogram-based tree learner."""
        import pandas as pd
        from lightgbm import LGBMClassifier
        
        # Convert to DataFrame for LightGBM
        X_df = pd.DataFrame(X)
        
        # Generate seeds
        np.random.seed(self.settings.random_seed)
        model_seeds = np.random.randint(0, 2**31, seeds).tolist()
        
        models = []
        for seed in model_seeds:
            model = LGBMClassifier(
                objective="binary",
                learning_rate=self.settings.learning_rate,
                num_leaves=self.settings.num_leaves,
                n_estimators=self.settings.n_estimators,
                subsample=self.settings.subsample,
                colsample_bytree=self.settings.colsample_bytree,
                random_state=seed,
                class_weight=self.settings.class_weight,
                boosting_type="gbdt",  # Histogram-based tree learner
                # GOSS off per constraints
                force_col_wise=True,  # Use histogram-based
            )
            model.fit(X_df, y)
            models.append(model)
        
        # Return bagged model
        from ..models.lightgbm import BaggedLightGBM
        return BaggedLightGBM(models=models)
    
    def _train_ffnn(self, X: np.ndarray, y: np.ndarray, seeds: int) -> Any:
        """Train small FFNN with focal loss."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch is required for FFNN training. Install with: pip install torch")
        
        # Generate seeds
        np.random.seed(self.settings.random_seed)
        model_seeds = np.random.randint(0, 2**31, seeds).tolist()
        
        models = []
        for seed in model_seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Small FFNN architecture
            model = SmallFFNN(input_dim=X.shape[1])
            
            # Focal loss with α=0.75, γ=2.0
            criterion = FocalLoss(alpha=0.75, gamma=2.0)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y)
            
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
            
            # Training loop
            model.train()
            for epoch in range(50):  # Small number of epochs
                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            models.append(model)
        
        return BaggedFFNN(models=models)


class SmallFFNN(nn.Module):
    """Small feedforward neural network for ransomware detection."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2),  # Binary classification
        )
    
    def forward(self, x):
        return self.network(x)


class FocalLoss(nn.Module):
    """Focal loss implementation for class imbalance."""
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class BaggedFFNN:
    """Bagged ensemble of FFNN models."""
    
    def __init__(self, models: list):
        self.models = models
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities by averaging ensemble predictions."""
        import torch
        
        all_probs = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                outputs = model(X_tensor)
                probs = torch.softmax(outputs, dim=1)[:, 1].numpy()
                all_probs.append(probs)
        
        return np.mean(np.vstack(all_probs), axis=0)
    
    def save(self, path):
        """Save the bagged model."""
        import joblib
        joblib.dump(self, path)
