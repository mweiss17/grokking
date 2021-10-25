import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

# Tensor of size 2
InputTensor = torch.Tensor
# Tensor of size 1
OutputTensor = torch.Tensor
# 0 = op, 1 = '='
NUM_SPECIAL_CHARS = 2

class ModularArithmetic:
    # TODO: Clean this up https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#IterableDataset
    def __init__(self, train_data_fraction: float = 0.5):
        self.p: int = 97
        self.train_data_fraction = train_data_fraction
        self.output_matrix = self.build_output_matrix()
        self.input_matrix = self.build_input_matrix()
        self.mask = self.build_mask()

    def build_output_matrix(self):
        matrix = torch.zeros(size=(self.p, self.p), dtype=int)
        for i in range(self.p):
            for j in range(self.p):
                matrix[i][j] = self.operation(i, j) + NUM_SPECIAL_CHARS
        return matrix

    def build_input_matrix(self):
        matrix = torch.zeros(size=(self.p, self.p, 4), dtype=int)
        for i in range(self.p):
            for j in range(self.p):
                # Increment both input values by two to make space for special tokens
                matrix[i][j][0] = i + NUM_SPECIAL_CHARS
                matrix[i][j][1] = 0  # special token for operation
                matrix[i][j][2] = j + NUM_SPECIAL_CHARS
                matrix[i][j][3] = 1  # special token for '='
        return matrix

    def build_mask(self):
        probs = torch.full(size=(self.p, self.p),
                        fill_value=self.train_data_fraction,
                        dtype=float)
        mask = torch.bernoulli(probs).bool()
        return mask

    def operation(self, a: int, b: int):
        return (a - b) % self.p


class ModularArithmeticDataset(Dataset):
    def __init__(self, ma: ModularArithmetic, train: bool):
        mask = ma.mask if train else ~ma.mask
        self.inputs = ma.input_matrix[mask]
        self.outputs = ma.output_matrix[mask]

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        return (self.inputs[idx], self.outputs[idx])

def collate_fn(samples: List[Tuple[InputTensor, OutputTensor]]):
    inputs = torch.stack([sample[0] for sample in samples])
    outputs = torch.stack([sample[1] for sample in samples])
    return (inputs, outputs)


if __name__ == '__main__':
    ma = ModularArithmetic()
    train_ds = ModularArithmeticDataset(ma, train=True)
    valid_ds = ModularArithmeticDataset(ma, train=False)
    batch_size = 512
    train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
    print(next(iter(train_dataloader)))