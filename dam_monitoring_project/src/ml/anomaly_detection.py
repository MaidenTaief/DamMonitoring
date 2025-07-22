#!/usr/bin/env python3
"""
Multi-modal Transformer for Dam Structural Health Monitoring
Anomaly detection using sensor fusion
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, List, Tuple, Optional
import psycopg2
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger(__name__)

class DamSensorDataset(Dataset):
    """Dataset for multi-modal dam sensor data"""
    
    def __init__(self, data: pd.DataFrame, sequence_length: int = 24, 
                 label_column: str = 'anomaly'):
        self.data = data
        self.sequence_length = sequence_length
        self.label_column = label_column
        
        # Get unique modalities
        self.modalities = data['sensor_modality'].unique()
        self.parameters = data['parameter_type'].unique()
        
        # Create sequences
        self.sequences = self._create_sequences()
    
    def _create_sequences(self) -> List[Dict]:
        """Create sequences for transformer input"""
        sequences = []
        for dam_id in self.data['dam_id'].unique():
            dam_data = self.data[self.data['dam_id'] == dam_id]
            dam_data = dam_data.sort_values('measurement_timestamp')
            timestamps = dam_data['measurement_timestamp'].unique()
            for i in range(len(timestamps) - self.sequence_length):
                seq_timestamps = timestamps[i:i + self.sequence_length]
                feature_matrix = []
                for ts in seq_timestamps:
                    ts_data = dam_data[dam_data['measurement_timestamp'] == ts]
                    features = []
                    for param in self.parameters:
                        param_value = ts_data[ts_data['parameter_type'] == param]['measurement_value'].values
                        if len(param_value) > 0:
                            features.append(param_value[0])
                        else:
                            features.append(0.0)
                    feature_matrix.append(features)
                if len(feature_matrix) == self.sequence_length:
                    sequence = {
                        'features': np.array(feature_matrix),
                        'label': 0,  # Default label, will be updated based on anomaly logic
                        'dam_id': dam_id,
                        'timestamp': seq_timestamps[-1]
                    }
                    sequences.append(sequence)
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return torch.FloatTensor(seq['features']), torch.FloatTensor([seq['label']])


class MultiModalTransformer(nn.Module):
    """Multi-modal transformer for dam health monitoring"""
    
    def __init__(self, n_features: int, n_heads: int = 8, n_layers: int = 4,
                 d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        
        # Input embedding
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.positional_encoding = self._create_positional_encoding(1000, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global pooling
        x = torch.mean(x, dim=1)
        
        # Classification head
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        
        return x


class DamAnomalyDetector:
    """Main class for dam anomaly detection"""
    
    def __init__(self, db_config: Dict[str, str]):
        self.db_config = db_config
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_sensor_data(self, days: int = 30) -> pd.DataFrame:
        """Load sensor data from database"""
        from sqlalchemy import create_engine
        engine = create_engine(
            f"postgresql://{self.db_config['user']}:{self.db_config['password']}@"
            f"{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        )
        query = f"""
        SELECT 
            d.dam_id,
            d.dam_name,
            sr.parameter_type,
            sr.sensor_modality,
            sr.measurement_value,
            sr.measurement_timestamp,
            sr.quality_code
        FROM sensor_readings sr
        JOIN monitoring_stations ms ON sr.station_id = ms.station_id
        JOIN dams d ON ms.dam_id = d.dam_id
        WHERE sr.measurement_timestamp > NOW() - INTERVAL '{days} days'
        ORDER BY d.dam_id, sr.measurement_timestamp
        """
        df = pd.read_sql(query, engine)
        df['anomaly'] = self._create_synthetic_anomalies(df)
        return df
    
    def _create_synthetic_anomalies(self, df: pd.DataFrame, anomaly_rate: float = 0.05) -> np.ndarray:
        """Create synthetic anomaly labels for training"""
        anomalies = np.zeros(len(df))
        
        # Define anomaly conditions
        for idx, row in df.iterrows():
            # Strain gauge anomalies
            if row['parameter_type'] == 'strain_gauge' and abs(row['measurement_value']) > 200:
                anomalies[idx] = 1
            
            # Accelerometer anomalies
            elif row['parameter_type'] == 'accelerometer' and abs(row['measurement_value']) > 0.1:
                anomalies[idx] = 1
            
            # Piezometer anomalies
            elif row['parameter_type'] == 'piezometer' and row['measurement_value'] > 200:
                anomalies[idx] = 1
            
            # Random anomalies
            elif np.random.random() < anomaly_rate:
                anomalies[idx] = 1
        
        return anomalies
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data for training"""
        # Create dataset
        dataset = DamSensorDataset(df, sequence_length=24)
        
        # Split data
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, 
                    epochs: int = 50) -> Dict:
        """Train the transformer model"""
        # Get feature dimensions
        sample_batch = next(iter(train_loader))
        n_features = sample_batch[0].shape[2]
        
        # Initialize model
        self.model = MultiModalTransformer(n_features=n_features).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # Training history
        history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        # Training loop
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()
                    
                    predicted = (outputs > 0.5).float()
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # Update scheduler
            scheduler.step(avg_val_loss)
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        return history
    
    def evaluate_model(self, test_loader: DataLoader) -> Dict:
        """Evaluate model performance"""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_features)
                predicted = (outputs > 0.5).float()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probs.extend(outputs.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)
        auc = roc_auc_score(all_labels, all_probs)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
        
        logger.info(f"Model Performance: {metrics}")
        
        return metrics
    
    def predict_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict anomalies in new data"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        dataset = DamSensorDataset(df, sequence_length=24, label_column=None)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_features, _ in dataloader:
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features)
                predictions.extend(outputs.cpu().numpy())
        
        # Add predictions to dataframe
        df['anomaly_score'] = predictions
        df['is_anomaly'] = df['anomaly_score'] > 0.5
        
        return df
    
    def save_model(self, path: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'n_features': self.model.n_features,
                'd_model': self.model.d_model
            }
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model = MultiModalTransformer(
            n_features=checkpoint['model_config']['n_features']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {path}")


def main():
    """Run anomaly detection training"""
    from dotenv import load_dotenv
    load_dotenv('config/.env')
    
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'database': os.getenv('DB_DATABASE', 'dam_monitoring'),
        'user': os.getenv('DB_USER', 'dam_user'),
        'password': os.getenv('DB_PASSWORD'),
        'port': os.getenv('DB_PORT', '5432')
    }
    
    # Initialize detector
    detector = DamAnomalyDetector(db_config)
    
    # Load data
    logger.info("Loading sensor data...")
    df = detector.load_sensor_data(days=30)
    
    # Prepare data
    logger.info("Preparing data...")
    train_loader, val_loader, test_loader = detector.prepare_data(df)
    
    # Train model
    logger.info("Training model...")
    history = detector.train_model(train_loader, val_loader, epochs=50)
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics = detector.evaluate_model(test_loader)
    
    # Save model
    detector.save_model('dam_monitoring_project/models/anomaly_detector.pth')
    
    # Save metrics
    with open('dam_monitoring_project/analysis/ml_metrics.json', 'w') as f:
        json.dump({
            'metrics': metrics,
            'history': history,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    os.makedirs('dam_monitoring_project/models', exist_ok=True)
    main() 