import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, Callable
from models.definitions.base_model import PyTorchModel
from src.utils.logs import logger
from src.utils import metrics_utils

class PyTorchTrainer(object):
    def __init__(self,
        model: PyTorchModel,
        optim: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device):
        self.model = model.to(device)
        self.optim = optim
        self.criterion = criterion
        self.device=device
        self._check_optim_net_aligned()
    def _check_optim_net_aligned(self):

        model_params = set(p for p in self.model.parameters())
        optim_params = set(p for g in self.optim.param_groups for p in g['params'])
        if not model_params.issubset(optim_params):
            raise ValueError("Optimizer missing some model parameters")


        #assert self.optim.param_groups[0]['params']\
        #== list(self.model.parameters())

    def fit(self, train_data: DataLoader, val_data: DataLoader,
        epochs: int=100,
        eval_every: int=10,
        clip_norm:float=1.0):
        for e in range(1,epochs+1):

            try:
                self.model.train()
                #X_train, y_train = permute_data(X_train, y_train)
                
                for X_batch, y_batch in train_data:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    output = self.model(X_batch).squeeze()
                    if output.shape[0] != y_batch.shape[0]:
                            raise ValueError(f"Output {output.shape} != Target {y_batch.shape}")

                    loss = self.criterion(output, y_batch)
                    self.optim.zero_grad()
                    loss.backward()
                    #print("after grad",loss)
                    if clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
                    self.optim.step()
                
                if e % eval_every == 0:

                    metrics={"R2_Score":metrics_utils.r2,"MAE":metrics_utils.mae}

                    self._evaluate(val_data, metrics, logger, e)
            
            except Exception as ex:
                logger.exception(f"Training failed at epoch {e}: {ex}")
                raise

    def _evaluate(self, val_data: DataLoader,
                  metrics: Optional[Dict[str, Callable]] = None,
                  logger=None, epoch: int = 0):
        self.model.eval()
        total_loss, n = 0.0, 0
        agg_metrics = {name: 0.0 for name in (metrics or {})}
        with torch.no_grad():
            for xb, yb in val_data:
                xb, yb = xb.to(self.device), yb.to(self.device)
                out = self.model(xb)
                total_loss += self.loss(out, yb).item() * xb.size(0)
                n += xb.size(0)
                if metrics:
                    for name, fn in metrics.items():
                        agg_metrics[name] += fn(out, yb) * xb.size(0)
        avg_loss = total_loss / max(n, 1)
        msg = f"epoch={epoch} val_loss={avg_loss:.4f}"
        if metrics:
            for name, total in agg_metrics.items():
                msg += f" {name}={total / max(n, 1):.4f}"
        (logger.info if logger else print)(msg)

                

