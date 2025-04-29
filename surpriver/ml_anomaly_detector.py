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

# Try to import TensorFlow with more robust error handling
TF_AVAILABLE = False
try:
    # Try to disable TensorFlow's OpenSSL dependency by setting environment variable
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
    
    # Try to work around OpenSSL issues
    os.environ["TF_USE_LEGACY_TENSORFLOW"] = "1"
    
    # First, try to import just the core tensorflow, avoiding keras
    import tensorflow as tf
    
    # Basic check if TensorFlow is actually functional
    test_tensor = tf.constant([1, 2, 3])
    
    # Check TensorFlow version and decide on implementation
    TF_AVAILABLE = True
    if hasattr(tf, '__version__'):
        print(f"TensorFlow version: {tf.__version__}")
    
except Exception as e:
    # Catch all TensorFlow import errors including OpenSSL issues
    print(f"TensorFlow not available or has compatibility issues: {str(e)}")
    print("TensorFlow-based models will be disabled.")
    TF_AVAILABLE = False

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


class SimpleTFAnomalyDetector(BaseAnomalyDetector):
    """Simple TensorFlow-based anomaly detector without using Keras"""
    
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, 
                 learning_rate=0.001, epochs=50, batch_size=32):
        super().__init__(name="SimpleTFDetector")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = {}
        
        if TF_AVAILABLE:
            try:
                self._build_model()
            except Exception as e:
                print(f"Error initializing TensorFlow model: {str(e)}")
                
    def _build_model(self):
        """Define simple TF variables without using Keras"""
        if not TF_AVAILABLE:
            return
        
        try:
            # Initialize weights and biases for a simple autoencoder
            # Encoder weights and biases
            initializer = tf.initializers.GlorotUniform()
            self.weights = {
                'encoder_h1': tf.Variable(initializer([self.input_dim, self.hidden_dim])),
                'encoder_b1': tf.Variable(tf.zeros([self.hidden_dim])),
                'encoder_h2': tf.Variable(initializer([self.hidden_dim, self.latent_dim])),
                'encoder_b2': tf.Variable(tf.zeros([self.latent_dim])),
                
                # Decoder weights and biases
                'decoder_h1': tf.Variable(initializer([self.latent_dim, self.hidden_dim])),
                'decoder_b1': tf.Variable(tf.zeros([self.hidden_dim])),
                'decoder_h2': tf.Variable(initializer([self.hidden_dim, self.input_dim])),
                'decoder_b2': tf.Variable(tf.zeros([self.input_dim])),
            }
        except Exception as e:
            print(f"Failed to initialize TensorFlow variables: {str(e)}")
            self.weights = {}
        
    def _encoder(self, x):
        """Simple encoder function"""
        layer_1 = tf.nn.relu(tf.matmul(x, self.weights['encoder_h1']) + self.weights['encoder_b1'])
        layer_2 = tf.nn.relu(tf.matmul(layer_1, self.weights['encoder_h2']) + self.weights['encoder_b2'])
        return layer_2
        
    def _decoder(self, x):
        """Simple decoder function"""
        layer_1 = tf.nn.relu(tf.matmul(x, self.weights['decoder_h1']) + self.weights['decoder_b1'])
        layer_2 = tf.matmul(layer_1, self.weights['decoder_h2']) + self.weights['decoder_b2']
        return layer_2
        
    def _autoencoder(self, x):
        """Full autoencoder function"""
        encoded = self._encoder(x)
        decoded = self._decoder(encoded)
        return decoded
        
    def fit(self, X):
        if not TF_AVAILABLE or not self.weights:
            print("TensorFlow is not available or model failed to initialize. Cannot train model.")
            return self
            
        try:
            X_scaled = self.preprocess(X)
            X_tensor = tf.convert_to_tensor(X_scaled, dtype=tf.float32)
            
            # Define optimizer
            optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
            
            # Split data into batches
            dataset = tf.data.Dataset.from_tensor_slices(X_tensor)
            dataset = dataset.shuffle(buffer_size=len(X_scaled)).batch(self.batch_size)
            
            # Training loop
            for epoch in range(self.epochs):
                for batch in dataset:
                    with tf.GradientTape() as tape:
                        # Forward pass
                        reconstructed = self._autoencoder(batch)
                        # Compute loss
                        loss = tf.reduce_mean(tf.square(reconstructed - batch))
                    
                    # Compute gradients
                    gradients = tape.gradient(loss, list(self.weights.values()))
                    # Apply gradients
                    optimizer.apply_gradients(zip(gradients, list(self.weights.values())))
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {loss.numpy():.4f}")
            
            return self
                
        except Exception as e:
            print(f"Error during TensorFlow model training: {str(e)}")
            return self
    
    def predict(self, X):
        if not TF_AVAILABLE or not self.weights:
            print("TensorFlow is not available or model failed to initialize. Cannot predict.")
            return np.zeros(X.shape[0])
            
        try:
            X_scaled = self.preprocess(X)
            X_tensor = tf.convert_to_tensor(X_scaled, dtype=tf.float32)
            
            # Get reconstructions
            reconstructed = self._autoencoder(X_tensor)
            
            # Calculate reconstruction error
            errors = tf.reduce_mean(tf.square(reconstructed - X_tensor), axis=1)
            
            # Invert so negative values indicate anomalies
            return -errors.numpy()
            
        except Exception as e:
            print(f"Error during TensorFlow prediction: {str(e)}")
            return np.zeros(X.shape[0])
    
    def is_available(self):
        return TF_AVAILABLE and len(self.weights) > 0


class XGBoostAnomalyDetector(BaseAnomalyDetector):
    """XGBoost-based anomaly detector"""
    
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=3):
        super().__init__(name="XGBoost")
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []
        
        if XGBOOST_AVAILABLE:
            self._build_model()
    
    def _build_model(self):
        """Define XGBoost model"""
        if not XGBOOST_AVAILABLE:
            return
            
        try:
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth
            )
        except Exception as e:
            print(f"Error initializing XGBoost model: {str(e)}")
            self.model = None
    
    def fit(self, X):
        if not XGBOOST_AVAILABLE:
            print("XGBoost is not available. Cannot train model.")
            return self
            
        try:
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
                
        except Exception as e:
            print(f"Error during XGBoost model training: {str(e)}")
            self.models = []
            
        return self
    
    def predict(self, X):
        if not XGBOOST_AVAILABLE or not self.models:
            print("XGBoost is not available or models failed to train. Cannot predict.")
            return np.zeros(X.shape[0])
            
        try:
            X_scaled = self.preprocess(X)
            n_features = X_scaled.shape[1]
            n_samples = X_scaled.shape[0]
            
            # Calculate reconstruction error for each feature
            reconstruction_errors = np.zeros((n_samples, n_features))
            
            for i in range(min(n_features, len(self.models))):
                X_pred = np.delete(X_scaled, i, axis=1)
                y_pred = self.models[i].predict(X_pred)
                reconstruction_errors[:, i] = np.square(X_scaled[:, i] - y_pred)
            
            # Mean reconstruction error across all features
            mean_reconstruction_error = np.mean(reconstruction_errors, axis=1)
            
            # Invert the errors so lower values indicate anomalies (to match Isolation Forest's output)
            anomaly_scores = -mean_reconstruction_error
            return anomaly_scores
            
        except Exception as e:
            print(f"Error during XGBoost prediction: {str(e)}")
            return np.zeros(X.shape[0])
    
    def is_available(self):
        return XGBOOST_AVAILABLE and len(self.models) > 0


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
            "SimpleTFDetector": 1.0,
            "XGBoost": 1.0
        }
        
    def create_models(self, input_dim):
        """Initialize all available models"""
        # Create models dictionary
        models = {
            "IsolationForest": IsolationForestDetector(n_estimators=100, contamination=0.1),
        }
        
        # Only add models that are available
        if PYTORCH_AVAILABLE:
            models["PyTorchAutoencoder"] = PyTorchAnomalyDetector(input_dim=input_dim)
            
        if TF_AVAILABLE:
            models["SimpleTFDetector"] = SimpleTFAnomalyDetector(input_dim=input_dim)
            
        if XGBOOST_AVAILABLE:
            models["XGBoost"] = XGBoostAnomalyDetector()
        
        # Filter out unavailable models and those that failed to initialize
        self.models = {name: model for name, model in models.items() if model.is_available()}
        
        if not self.models:
            print("No ML models are available. Falling back to Isolation Forest.")
            self.models = {"IsolationForest": IsolationForestDetector(n_estimators=100, contamination=0.1)}
        
        # Check if the requested method is available
        if self.method not in ['ensemble', 'best'] and self.method not in self.models:
            print(f"Warning: Requested method '{self.method}' is not available. Falling back to 'isolation_forest'.")
            self.method = 'isolation_forest' if 'IsolationForest' in self.models else 'ensemble'
            
        print(f"Available models: {', '.join(self.models.keys())}")
    
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
        for name, model in list(self.models.items()):  # Use list() to avoid dictionary changed size during iteration
            print(f"  - Training {name}...")
            try:
                model.fit(features)
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                # Remove the model if training failed
                self.models.pop(name, None)
        
        # If all models failed, fall back to Isolation Forest
        if not self.models:
            print("All models failed to train. Falling back to Isolation Forest.")
            model = IsolationForestDetector(n_estimators=100, contamination=0.1)
            model.fit(features)
            self.models = {"IsolationForest": model}
        
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
        # If no models, return zeros
        if not self.models:
            print("No models available. Returning zeros.")
            return np.zeros(features.shape[0])
        
        # Special case: if only one model is available, use it
        if len(self.models) == 1:
            name, model = next(iter(self.models.items()))
            print(f"Using only available model: {name}")
            return model.predict(features)
        
        if self.method == 'ensemble':
            # Weighted average of all model predictions
            weighted_scores = np.zeros(features.shape[0])
            total_weight = 0
            
            for name, model in self.models.items():
                try:
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
                except Exception as e:
                    print(f"Error using {name} in ensemble: {str(e)}")
            
            return weighted_scores / total_weight if total_weight > 0 else weighted_scores
            
        elif self.method == 'best':
            # Find the "best" model (in this case, the one with the widest range of scores)
            best_model = None
            best_range = -1
            
            for name, model in self.models.items():
                try:
                    predictions = model.predict(features)
                    pred_range = np.max(predictions) - np.min(predictions)
                    
                    if pred_range > best_range:
                        best_range = pred_range
                        best_model = model
                except Exception as e:
                    print(f"Error evaluating {name} for 'best' selection: {str(e)}")
            
            if best_model:
                return best_model.predict(features)
            else:
                # Fallback to isolation forest if no best model could be determined
                return self.models.get("IsolationForest", list(self.models.values())[0]).predict(features)
        
        elif self.method == 'isolation_forest' and 'IsolationForest' in self.models:
            # Use Isolation Forest specifically
            return self.models['IsolationForest'].predict(features)
            
        elif self.method == 'tensorflow' and 'SimpleTFDetector' in self.models:
            # Use TensorFlow specifically
            return self.models['SimpleTFDetector'].predict(features)
            
        else:
            # Try to use the specified model
            if self.method in self.models:
                return self.models[self.method].predict(features)
            else:
                # Fallback to isolation forest or first available model
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