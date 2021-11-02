import random
import torch
from grokking.dataset import ModularArithmetic, ModularArithmeticDataset

seed = 1

random.seed(seed)
torch.manual_seed(seed)

ma = ModularArithmetic()
valid_dataset = ModularArithmeticDataset(ma, train=False)
train_dataset = ModularArithmeticDataset(ma, train=True)
train_dataset.save()
valid_dataset.save()