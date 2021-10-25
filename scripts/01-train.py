import torch
import sys
from torch.utils.data import Dataset, DataLoader
from grokking.dataset import ModularArithmetic, ModularArithmeticDataset, collate_fn
from speedrun import BaseExperiment, WandBMixin, IOMixin
from grokking.model import TransformerModel

class Trainer(BaseExperiment, WandBMixin, IOMixin):
    WANDB_PROJECT = "mila-grokking"
    WANDB_ENTITY = "mweiss10"

    def __init__(self):
        super(Trainer, self).__init__()
        self.auto_setup()
        self.train_dataloader, self.valid_dataloader = self._build_dataset()
        self.device = torch.device('cpu')
        self.model = self._build_model()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.get("lr"), betas=(0.9, 0.98), weight_decay=self.get("weight_decay"))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)


    def _build_dataset(self):
        ma = ModularArithmetic()
        train_ds = ModularArithmeticDataset(ma, train=True)
        valid_ds = ModularArithmeticDataset(ma, train=False)
        train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=self.get("batch_size"))
        valid_dataloader = DataLoader(valid_ds, shuffle=True, collate_fn=collate_fn, batch_size=self.get("batch_size"))
        return train_dataloader, valid_dataloader

    def _build_model(self):
        model = TransformerModel(self.get("ntokens"), self.get("emsize"), self.get("nhead"), self.get("d_hid"), self.get("nlayers"), self.get("dropout")).to(self.device)
        return model

    def run(self):
        for step, (inputs, outputs) in enumerate(self.train_dataloader):
            # Forward
            preds = self.model(inputs, torch.tensor([0]))

            # Compute loss
            self.model.loss_fn(preds, outputs)

            # Optimize
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Record metrics

            # Checkpoint

    def save_checkpoint(self):
        data = {"model": self.model.state_dict(), "optim": self.optimizer.state_dict()}
        checkpoint_path = f"{self.experiment_directory}/Weights/model-{self.step}.pt"
        print(f"checkpointing the model to {checkpoint_path}")
        torch.save(data, checkpoint_path)

    def load_checkpoint(self):
        checkpoint_data = torch.load(self.get("checkpoint_path"))
        self.model.load_state_dict(checkpoint_data["model"])
        self.optimizer.load_state_dict(checkpoint_data["optim"])


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv.append("experiments/arithmetic-test")
        sys.argv.append("--inherit")
        sys.argv.append("templates/arithmetic")
    Trainer().run()
