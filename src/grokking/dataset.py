import torch
from torch.utils.data import Dataset


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
                matrix[i][j] = self.operation(i, j)
        return matrix

    def build_input_matrix(self):
        matrix = torch.zeros(size=(self.p, self.p, 2), dtype=int)
        for i in range(self.p):
            for j in range(self.p):
                matrix[i][j][0] = i
                matrix[i][j][1] = j
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




if __name__ == '__main__':
    ma = ModularArithmetic()
    train_ds = ModularArithmeticDataset(ma, train=True)
    valid_ds = ModularArithmeticDataset(ma, train=False)
    print("ouh")