import torch
import sys
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from grokking.dataset import ModularArithmetic, ModularArithmeticDataset, collate_fn
from speedrun import BaseExperiment, WandBMixin, IOMixin
from grokking.model import TransformerModel

class Trainer(BaseExperiment, WandBMixin, IOMixin):
    WANDB_PROJECT = "mila-grokking"
    WANDB_ENTITY = "mweiss10"

    def __init__(self):
        super(Trainer, self).__init__()
        self.auto_setup()

        # In case we want to override the default wandb project
        self.WANDB_PROJECT = self.get("WANDB_PROJECT", self.WANDB_PROJECT)
        self.WANDB_ENTITY = self.get("WANDB_PROJECT", self.WANDB_ENTITY)

        self.set_seeds(self.get("seed"))
        if self.get("use_wandb"):
            self.initialize_wandb()
        self.train_dataloader, self.valid_dataloader = self._build_dataset()
        self.device = torch.device('cpu')
        self._build_model()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)


    def set_seeds(self, seed):
        random.seed(seed)
        torch.manual_seed(seed)

    def _build_dataset(self):
        train_ds = torch.load(f"{self.get('dataset_path')}/{self.get('dataset_name')}_train.pt")
        valid_ds = torch.load(f"{self.get('dataset_path')}/{self.get('dataset_name')}_valid.pt")

        train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=self.get("batch_size"))
        valid_dataloader = DataLoader(valid_ds, shuffle=True, collate_fn=collate_fn, batch_size=self.get("batch_size"))
        return train_dataloader, valid_dataloader

    def _build_model(self):
        self.model = TransformerModel(self.get("ntokens"), self.get("emsize"), self.get("nhead"), self.get("d_hid"), self.get("nlayers"), self.get("dropout")).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.get("lr"), betas=(0.9, 0.98), weight_decay=self.get("weight_decay"))
        if self.get("checkpoint_path"):
            self.load_checkpoint()

    def compute_accuracy(self, preds, labels):

        correct = preds.argmax(1).eq(labels).sum() / labels.size()[0]
        return correct
    
    def test(self, split="valid"):
        self.model.eval()
        if split == "valid":
            dataloader = self.valid_dataloader
        else:
            dataloader = self.train_dataloader

        total_correct = 0
        total_ex = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs[:, :5].T
                preds = self.model(inputs)
                correct = self.compute_accuracy(preds, labels)
                total_correct += correct * labels.size(0)
                total_ex += labels.size(0)
        self.model.train()
        return total_correct / total_ex

    def run(self):
        for epoch in range(self.get("n_epochs")):
            for step, (inputs, labels) in enumerate(self.train_dataloader):
                # Forward
                #manual adjustment to inputs for now
                inputs = inputs[:, :5].T
                preds = self.model(inputs)
                
                if self.step % self.get("print_every") == 0:
                    correct = self.compute_accuracy(preds, labels)
                    val_correct = self.test(split="valid")
                    train_correct = self.test(split="train")

                # Compute loss
                loss = self.model.loss_fn(preds, labels)
                loss.backward()

                # Optimize
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.next_step()

                # Record metrics
                if self.step % self.get("print_every") == 0:
                    if self.get("use_wandb"):
                        self.wandb_log(**{"loss": loss.detach(), "step": self.step, "train batch accuracy": correct, "test accuracy": val_correct, "train global accuracy": train_correct})
                    else:
                        print(f'"loss": {loss.detach()}, "epoch":{epoch}, "step": {self.step}, "train batch accuracy": {correct}, "test global accuracy": {val_correct}, train global accuracy: {train_correct}')
                        
                # Checkpoint
                if self.get("save_every") > 0 and self.step % self.get("save_every") == 0:
                    self.save_checkpoint()

            self.next_epoch()

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
