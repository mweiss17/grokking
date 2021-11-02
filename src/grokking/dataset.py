import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Callable

# Tensor of size 2
InputTensor = torch.Tensor
# Tensor of size 1
OutputTensor = torch.Tensor
# 0 = op, 1 = '='
NUM_SPECIAL_CHARS = 3

class ModularArithmetic:
    # TODO: Clean this up https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html#IterableDataset
    def __init__(self, train_data_fraction: float = 0.5, operation: Callable = None, name: str = "subtraction_mod_97"):
        self.p: int = 97
        self.name = name
        self.train_data_fraction = train_data_fraction
        self.operation = operation if operation is not None else lambda x, y: (x - y) % self.p
        self.input_matrix = self.build_input_matrix()
        self.output_matrix = self.build_output_matrix()
        self.mask = self.build_mask()

    def build_output_matrix(self):
        matrix = torch.zeros(size=(self.p, self.p), dtype=int)
        for i in range(self.p):
            for j in range(self.p):
                matrix[i][j] = self.operation(i, j) + NUM_SPECIAL_CHARS
        return matrix

    def build_input_matrix(self):
        matrix = torch.zeros(size=(self.p, self.p, 6), dtype=int)
        for i in range(self.p):
            for j in range(self.p):
                # Increment both input values by two to make space for special tokens
                matrix[i][j][0] = 0  # Start of Sequence token
                matrix[i][j][1] = i + NUM_SPECIAL_CHARS
                matrix[i][j][2] = 1  # special token for operation
                matrix[i][j][3] = j + NUM_SPECIAL_CHARS
                matrix[i][j][4] = 2  # special token for '='
                matrix[i][j][5] = self.operation(i, j) + NUM_SPECIAL_CHARS
        return matrix

    def build_mask(self):
        probs = torch.full(size=(self.p, self.p),
                        fill_value=self.train_data_fraction,
                        dtype=float)
        mask = torch.bernoulli(probs).bool()
        return mask



class ModularArithmeticDataset(Dataset):
    def __init__(self, ma: ModularArithmetic, train: bool):
        self.split = "train" if train else "valid"
        self.name = ma.name
        mask = ma.mask if train else ~ma.mask
        self.inputs = ma.input_matrix[mask]
        self.outputs = ma.output_matrix[mask]

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, idx):
        return (self.inputs[idx], self.outputs[idx])

    def save(self):
        torch.save(self, f"datasets/{self.name}_{self.split}.pt")


def collate_fn(samples: List[Tuple[InputTensor, OutputTensor]]):
    inputs = torch.stack([sample[0] for sample in samples])
    outputs = torch.stack([sample[1] for sample in samples])
    return (inputs, outputs)


# if __name__ == '__main__':
#     ma = ModularArithmetic()
#     train_ds = ModularArithmeticDataset(ma, train=True)
#     valid_ds = ModularArithmeticDataset(ma, train=False)
#     batch_size = 512
#     train_dataloader = DataLoader(train_ds, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
#     print(next(iter(train_dataloader)))