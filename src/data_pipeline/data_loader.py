from dataclasses import dataclass
from pathlib import Path
import torch
import pandas as pd
from typing import Tuple
from torch.utils.data import TensorDataset, DataLoader

@dataclass
class DatasetConfig:
    target_col: str
    val_ratio: float = 0.2
    seed: int = 42
    batch_size: int = 64

class DataReader:
    def __init__(self,root:Path)->None:
        self.root=root

    def read(self,name:str)-> pd.DataFrame:
        path = self.root / name
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        df=pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Empty data file: {path}")
        return df
    
class DataMaker(DataReader):
    def __init__(self, root)->None:
        super().__init__(root)

    def make_data(self,name:str,
                  cfg:DatasetConfig)->Tuple[DataLoader,DataLoader]:

        data=self.read(name)

        train_df,val_df=self.split_shuffle(data,cfg)

        X,y = self.convert_to_tensor(train_df,cfg.target_col)
        X_val,y_val=self.convert_to_tensor(val_df,cfg.target_col)

        return (DataLoader(TensorDataset(X, y), batch_size=cfg.batch_size, shuffle=True),
                DataLoader(TensorDataset(X_val, y_val), batch_size=cfg.batch_size))
    
    def split_shuffle(self,df: pd.DataFrame, cfg: DatasetConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if cfg.target_col not in df.columns:
            raise KeyError(f"Target column '{cfg.target_col}' not in data.")
        df = df.sample(frac=1.0, random_state=cfg.seed)
        n_val = max(1, int(len(df) * cfg.val_ratio))
        return df.iloc[n_val:], df.iloc[:n_val]
    
    def convert_to_tensor(self,df:pd.DataFrame,target_col:str)->Tuple[torch.Tensor,torch.Tensor]:
        y = torch.tensor(df[target_col].values, dtype=torch.float32)
        X = torch.tensor(df.drop(columns=[target_col]).values, dtype=torch.float32)
        return X, y