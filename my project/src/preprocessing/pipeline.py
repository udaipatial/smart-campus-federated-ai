"""
Preprocessing and Feature Engineering Pipeline
Handles cyclical temporal encoding, feature normalization, and PyTorch dataset preparation.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import config

class FeatureScaler:
    """Encapsulates standard scaling for multidimensional energy metrics."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def fit(self, X: np.ndarray):
        self.scaler.fit(X)
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Scaler has not been fitted yet.")
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer temporal cyclical features (sin/cos encoding of hour and day of week).
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
    hour = df["timestamp"].dt.hour
    day = df["timestamp"].dt.dayofweek
    
    # Cyclical sin/cos encoding
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["day_sin"] = np.sin(2 * np.pi * day / 7.0)
    df["day_cos"] = np.cos(2 * np.pi * day / 7.0)
    
    return df

def preprocess_dataframe(
    df: pd.DataFrame, 
    scaler: FeatureScaler = None, 
    fit_scaler: bool = True
) -> tuple[np.ndarray, np.ndarray, FeatureScaler, pd.DataFrame]:
    """
    Preprocess a building dataframe into feature matrices (X) and anomaly labels (y).
    """
    df_feat = engineer_features(df)
    
    X_raw = df_feat[config.FEATURE_COLUMNS].values.astype(np.float32)
    y = df_feat["is_anomaly"].values.astype(np.int64)
    
    if scaler is None:
        scaler = FeatureScaler()
        
    if fit_scaler:
        X_scaled = scaler.fit_transform(X_raw)
    else:
        X_scaled = scaler.transform(X_raw)
        
    return X_scaled, y, scaler, df_feat

def load_and_preprocess_building_data(
    building_id: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> dict:
    """
    Loads local building data, splits chronologically into train, val, test,
    and scales features strictly using the training set statistics to avoid data leakage.
    """
    file_path = config.DATA_DIR / f"{building_id}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Building data file not found at {file_path}")
        
    df = pd.read_csv(file_path)
    df_feat = engineer_features(df)
    
    total_len = len(df_feat)
    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))
    
    df_train = df_feat.iloc[:train_end].copy()
    df_val = df_feat.iloc[train_end:val_end].copy()
    df_test = df_feat.iloc[val_end:].copy()
    
    # Separate normal data for training the baseline normal representation
    # (Unsupervised / semi-supervised paradigm standard in energy anomaly detection)
    df_train_normal = df_train[df_train["is_anomaly"] == 0]
    
    scaler = FeatureScaler()
    scaler.fit(df_train_normal[config.FEATURE_COLUMNS].values.astype(np.float32))
    
    X_train = scaler.transform(df_train_normal[config.FEATURE_COLUMNS].values.astype(np.float32))
    y_train = df_train_normal["is_anomaly"].values.astype(np.int64)
    
    X_val = scaler.transform(df_val[config.FEATURE_COLUMNS].values.astype(np.float32))
    y_val = df_val["is_anomaly"].values.astype(np.int64)
    
    X_test = scaler.transform(df_test[config.FEATURE_COLUMNS].values.astype(np.float32))
    y_test = df_test["is_anomaly"].values.astype(np.int64)
    
    return {
        "building_id": building_id,
        "scaler": scaler,
        "train": {"X": X_train, "y": y_train, "df": df_train_normal},
        "val": {"X": X_val, "y": y_val, "df": df_val},
        "test": {"X": X_test, "y": y_test, "df": df_test},
        "full_df": df_feat
    }

def create_pytorch_dataloaders(
    X_train: np.ndarray,
    X_val: np.ndarray = None,
    X_test: np.ndarray = None,
    batch_size: int = 32
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch DataLoaders for train, val, and test partitions."""
    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    # For autoencoders, input and target are both X
    train_dataset = TensorDataset(train_tensor, train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if X_val is not None:
        val_tensor = torch.tensor(X_val, dtype=torch.float32)
        val_dataset = TensorDataset(val_tensor, val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
    test_loader = None
    if X_test is not None:
        test_tensor = torch.tensor(X_test, dtype=torch.float32)
        test_dataset = TensorDataset(test_tensor, test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
    return train_loader, val_loader, test_loader
