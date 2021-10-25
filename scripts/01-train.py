import torch
from torch.utils.data import Dataset, DataLoader
from grokking.dataset import ModularArithmetic, ModularArithmeticDataset, collate_fn
from speedrun import BaseExperiment, WandBMixing, IOMixing


class Trainer(BaseExperiment, WandBMixin, IOMixin):
    WANDB_PROJECT = "grokking"
    WANDB_ENTITY = "mweiss10"

    def __init__(self):
        super(Trainer, self).__init__()
        self.auto_setup()
        self.train_dataloader, self.valid_dataloader = self._build()

    def _build(self):
        ma = ModularArithmetic()
        train_ds = ModularArithmeticDataset(ma, train=True)
        valid_ds = ModularArithmeticDataset(ma, train=False)
        batch_size = 512
        train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
        valid_dataloader = DataLoader(valid_ds, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
        return train_dataloader, valid_dataloader

#
for step, (inputs, outputs) in enumerate(iter(train_dataloader)):

