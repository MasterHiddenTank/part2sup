"""
Advanced Machine Learning Anomaly Detection System for Surpriver
This module implements various ML frameworks for anomaly detection:
- PyTorch (Neural Network-based anomaly detection)
- TensorFlow (Autoencoder-based anomaly detection)
- XGBoost (Supervised anomaly detection)
- Scikit-learn (Enhanced isolation forest and other models)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import ML frameworks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. PyTorch-based models will be disabled.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Dense, Input, Dropout, LSTM
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. TensorFlow-based models will be disabled.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. XGBoost-based models will be disabled.")

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("DeepSpeed not available. DeepSpeed optimizations will be disabled.")

class BaseAnomalyDetector:
    """Base class for all anomaly detection models"""
    
    def __init__(self, name="BaseModel"):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        
    def preprocess(self, X):
        """Scale features"""
        return self.scaler.transform(X)
        
    def fit(self, X):
        """Train the model"""
        raise NotImplementedError("Subclasses must implement fit method")
        
    def predict(self, X):
        """Return anomaly scores"""
        raise NotImplementedError("Subclasses must implement predict method")
    
    def is_available(self):
        """Check if required dependencies are available"""
        return True


class IsolationForestDetector(BaseAnomalyDetector):
    """Enhanced Isolation Forest from scikit-learn"""
    
    def __init__(self, n_estimators=100, contamination=0.1, random_state=0):
        super().__init__(name="IsolationForest")
        from sklearn.ensemble import IsolationForest
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        )
        
    def fit(self, X):
        X_scaled = self.preprocess(X)
        self.model.fit(X_scaled)
        return self
        
    def predict(self, X):
        X_scaled = self.preprocess(X)
        # Return decision function (lower values indicate more anomalous instances)
        return self.model.decision_function(X_scaled)


class PyTorchAnomalyDetector(BaseAnomalyDetector):
    """Neural Network-based anomaly detector using PyTorch"""
    
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, 
                 learning_rate=0.001, epochs=50, batch_size=32):
        super().__init__(name="PyTorchAutoencoder")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        if PYTORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._build_model()
        
    def _build_model(self):
        """Define autoencoder architecture"""
        class Autoencoder(nn.Module):
            def __init__(self, input_dim, hidden_dim, latent_dim):
                super(Autoencoder, self).__init__()
                # Encoder
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, latent_dim),
                    nn.ReLU()
                )
                # Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, input_dim)
                )
            
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded
        
        self.model = Autoencoder(self.input_dim, self.hidden_dim, self.latent_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
    def fit(self, X):
        if not PYTORCH_AVAILABLE:
            print("PyTorch is not available. Cannot train model.")
            return self
            
        X_scaled = self.preprocess(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for data, _ in dataloader:
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, data)
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(dataloader):.4f}')
                
        return self
    
    def predict(self, X):
        if not PYTORCH_AVAILABLE:
            print("PyTorch is not available. Cannot predict.")
            return np.zeros(X.shape[0])
            
        X_scaled = self.preprocess(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            # Calculate reconstruction error (MSE) for each sample
            reconstruction_errors = torch.mean((outputs - X_tensor)**2, dim=1).cpu().numpy()
            
        # Invert the errors so lower values indicate anomalies (to match Isolation Forest's output)
        anomaly_scores = -reconstruction_errors
        return anomaly_scores
    
    def is_available(self):
        return PYTORCH_AVAILABLE


class TensorFlowAnomalyDetector(BaseAnomalyDetector):
    """Autoencoder-based anomaly detector using TensorFlow"""
    
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, 
                 learning_rate=0.001, epochs=50, batch_size=32):
        super().__init__(name="TFAutoencoder")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        
        if TF_AVAILABLE:
            self._build_model()
    
    def _build_model(self):
        """Define autoencoder architecture"""
        # Encoder
        input_layer = Input(shape=(self.input_dim,))
        encoded = Dense(self.hidden_dim, activation='relu')(input_layer)
        encoded = Dense(self.latent_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(self.hidden_dim, activation='relu')(encoded)
        decoded = Dense(self.input_dim, activation='linear')(decoded)
        
        # Autoencoder model
        self.model = Model(inputs=input_layer, outputs=decoded)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
                          loss='mse')
    
    def fit(self, X):
        if not TF_AVAILABLE:
            print("TensorFlow is not available. Cannot train model.")
            return self
            
        X_scaled = self.preprocess(X)
        
        self.model.fit(
            X_scaled, X_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=0
        )
        return self
    
    def predict(self, X):
        if not TF_AVAILABLE:
            print("TensorFlow is not available. Cannot predict.")
            return np.zeros(X.shape[0])
            
        X_scaled = self.preprocess(X)
        
        # Get reconstructions
        reconstructions = self.model.predict(X_scaled)
        # Calculate MSE for each sample
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        # Invert the errors so lower values indicate anomalies (to match Isolation Forest's output)
        anomaly_scores = -mse
        return anomaly_scores
    
    def is_available(self):
        return TF_AVAILABLE


class XGBoostAnomalyDetector(BaseAnomalyDetector):
    """XGBoost-based anomaly detector"""
    
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=3):
        super().__init__(name="XGBoost")
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
        if XGBOOST_AVAILABLE:
            self._build_model()
    
    def _build_model(self):
        """Define XGBoost model"""
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth
        )
    
    def fit(self, X):
        if not XGBOOST_AVAILABLE:
            print("XGBoost is not available. Cannot train model.")
            return self
            
        X_scaled = self.preprocess(X)
        
        # For unsupervised anomaly detection, we'll predict each feature from the others
        # We'll use the mean reconstruction error as the anomaly score
        n_features = X_scaled.shape[1]
        self.models = []
        
        for i in range(n_features):
            # Prepare data
            X_train = np.delete(X_scaled, i, axis=1)
            y_train = X_scaled[:, i]
            
            # Train model to predict feature i from the other features
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth
            )
            model.fit(X_train, y_train)
            self.models.append(model)
        
        return self
    
    def predict(self, X):
        if not XGBOOST_AVAILABLE:
            print("XGBoost is not available. Cannot predict.")
            return np.zeros(X.shape[0])
            
        X_scaled = self.preprocess(X)
        n_features = X_scaled.shape[1]
        n_samples = X_scaled.shape[0]
        
        # Calculate reconstruction error for each feature
        reconstruction_errors = np.zeros((n_samples, n_features))
        
        for i in range(n_features):
            X_pred = np.delete(X_scaled, i, axis=1)
            y_pred = self.models[i].predict(X_pred)
            reconstruction_errors[:, i] = np.square(X_scaled[:, i] - y_pred)
        
        # Mean reconstruction error across all features
        mean_reconstruction_error = np.mean(reconstruction_errors, axis=1)
        
        # Invert the errors so lower values indicate anomalies (to match Isolation Forest's output)
        anomaly_scores = -mean_reconstruction_error
        return anomaly_scores
    
    def is_available(self):
        return XGBOOST_AVAILABLE


class MLAnomalyDetectionSystem:
    """
    Advanced anomaly detection system that combines multiple ML models
    """
    
    def __init__(self, method='ensemble', model_weights=None):
        """
        Initialize the anomaly detection system
        
        Parameters:
        -----------
        method : str
            Method to combine models ('ensemble', 'best', 'pytorch', 'tensorflow', 'xgboost', or 'isolation_forest')
        model_weights : dict
            Weights for each model in the ensemble
        """
        self.method = method
        self.models = {}
        self.model_weights = model_weights or {
            "IsolationForest": 1.0,
            "PyTorchAutoencoder": 1.0,
            "TFAutoencoder": 1.0,
            "XGBoost": 1.0
        }
        
    def create_models(self, input_dim):
        """Initialize all available models"""
        # Create models dictionary
        models = {
            "IsolationForest": IsolationForestDetector(n_estimators=100, contamination=0.1),
            "PyTorchAutoencoder": PyTorchAnomalyDetector(input_dim=input_dim),
            "TFAutoencoder": TensorFlowAnomalyDetector(input_dim=input_dim),
            "XGBoost": XGBoostAnomalyDetector()
        }
        
        # Filter out unavailable models
        self.models = {name: model for name, model in models.items() if model.is_available()}
        
        if not self.models:
            raise ValueError("No models are available. Please install at least one of the required libraries.")
        
        if self.method not in ['ensemble', 'best'] and self.method not in self.models:
            print(f"Warning: Requested method '{self.method}' is not available. Falling back to 'ensemble'.")
            self.method = 'ensemble'
    
    def fit(self, features):
        """
        Train all models on the given features
        
        Parameters:
        -----------
        features : numpy.ndarray
            Feature matrix of shape (n_samples, n_features)
        """
        input_dim = features.shape[1]
        self.create_models(input_dim)
        
        print(f"Training {len(self.models)} anomaly detection models:")
        for name, model in self.models.items():
            print(f"  - Training {name}...")
            model.fit(features)
        
        return self
    
    def predict(self, features):
        """
        Generate anomaly scores for the given features
        
        Parameters:
        -----------
        features : numpy.ndarray
            Feature matrix of shape (n_samples, n_features)
            
        Returns:
        --------
        numpy.ndarray
            Anomaly scores (lower values indicate more anomalous instances)
        """
        if self.method == 'ensemble':
            # Weighted average of all model predictions
            weighted_scores = np.zeros(features.shape[0])
            total_weight = 0
            
            for name, model in self.models.items():
                weight = self.model_weights.get(name, 1.0)
                predictions = model.predict(features)
                
                # Normalize predictions to [0, 1] range
                min_pred = np.min(predictions)
                max_pred = np.max(predictions)
                if max_pred > min_pred:
                    normalized_predictions = (predictions - min_pred) / (max_pred - min_pred)
                else:
                    normalized_predictions = np.zeros_like(predictions)
                
                weighted_scores += weight * normalized_predictions
                total_weight += weight
            
            return weighted_scores / total_weight if total_weight > 0 else weighted_scores
            
        elif self.method == 'best':
            # Find the "best" model (in this case, the one with the widest range of scores)
            best_model = None
            best_range = -1
            
            for name, model in self.models.items():
                predictions = model.predict(features)
                pred_range = np.max(predictions) - np.min(predictions)
                
                if pred_range > best_range:
                    best_range = pred_range
                    best_model = model
            
            if best_model:
                return best_model.predict(features)
            else:
                # Fallback to isolation forest if no best model could be determined
                return self.models.get("IsolationForest", list(self.models.values())[0]).predict(features)
        
        else:
            # Use a specific model
            if self.method in self.models:
                return self.models[self.method].predict(features)
            else:
                # Fallback to isolation forest
                return self.models.get("IsolationForest", list(self.models.values())[0]).predict(features)
    
    def get_available_methods(self):
        """
        Get a list of available anomaly detection methods
        
        Returns:
        --------
        list
            List of available methods
        """
        methods = ['ensemble', 'best']
        methods.extend(list(self.models.keys()))
        return methods 