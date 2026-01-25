from train.trainer import PyTorchTrainer
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from dataclasses import dataclass
from src.utils.logs import logger
from src.utils import config_utils
from src.models.definitions.house_model import HousePricesModel
from src.data_pipeline import data_loader


@dataclass
class TrainConfig:
    lr: float
    momentum: float
    epochs: int
    seed: int = 42
    num_workers: int = 4


class Train:
    def __init__(self,model:HousePricesModel,optimizer:torch.optim.Optimizer,
                 loss_fn:torch.nn.MSELoss,epochs:int, device: torch.device) -> None:
        self.model=model
        self.optimizer=optimizer
        self.loss_fn=loss_fn
        self.epochs=epochs
        self.device=device

    def fit(self,train_data: DataLoader,val_data: DataLoader)-> None:
        trainer = PyTorchTrainer(self.model, self.optimizer, self.loss_fn,self.device)
        trainer.fit(train_data, val_data, epochs=self.epochs)


class CheckpointManager:
    @staticmethod
    def save(path: Path, model: HousePricesModel, optimizer: torch.optim.Optimizer, train_cfg: dict):
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "version": 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_config": train_cfg,
            "model_class": model.__class__.__name__,
        }, path)



#if __name__ == "__main__":

def training_pipeline()-> None:
    # Initializing model path and name
    try:
        path_cfg = config_utils.paths_config()
        PATH = Path(path_cfg['model']['path'])
        NAME=path_cfg['model']['name']
    except KeyError as e:
        logger.error(f"Missing config key: {e}")
        raise KeyError(f"Missing config key: {e}")
    
    # Loading, and spliting data 
    data_maker_obj=data_loader.DataMaker(Path(path_cfg['process_data']['path']))
    logger.info("Structure Data Loading.")
    data_cfg = data_loader.DatasetConfig(**config_utils.data_config()['dataconfig'])
    train_data, val_data=data_maker_obj.make_data(path_cfg['process_data']['name'],data_cfg)
    logger.info("Structure Data Loaded.")


    # Hyperparameters

    train_cfg=config_utils.train_config()['training']
    cfg = TrainConfig(
        lr=train_cfg['learning_rate'],
        momentum=train_cfg['momentum'],
        epochs=train_cfg['num_epochs'],
        seed=train_cfg['seed']
    )

    torch.manual_seed(cfg.seed)



    # Initialize model, optimizer, and loss function

    model = HousePricesModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr,
                                momentum=cfg.momentum)
    criterion = torch.nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Model Build.")
    # Train the model
    trainer_object=Train(model=model, optimizer=optimizer,
                            loss_fn=criterion, epochs=cfg.epochs,device=device)
    logger.info("Training started...")
    trainer_object.fit(train_data,val_data)
    logger.info("Training finished.")
    # Saving the trained model


    logger.info("Saving train model...")
    CheckpointManager.save(Path(PATH)/NAME, model, optimizer, train_cfg)
    logger.info("Model saved.")
