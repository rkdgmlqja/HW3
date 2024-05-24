import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    """Shakespeare dataset

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
        You need this dictionary to generate characters.
        2) Make list of character indices using the dictionary
        3) Split the data into chunks of sequence length 30.
        You should create targets appropriately.
    """

    def __init__(self, input_file):
        with open(input_file, 'r') as f:
            self.text = f.read()

        self.chars = sorted(list(set(self.text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        self.data = [self.char_to_idx[ch] for ch in self.text]

        self.seq_length = 30
        self.num_sequences = (len(self.data) - 1) // self.seq_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length

        input_seq = torch.tensor(self.data[start_idx:end_idx], dtype=torch.long)
        target_seq = torch.tensor(self.data[start_idx + 1:end_idx + 1], dtype=torch.long)

        return input_seq, target_seq

if __name__ == '__main__':
    # Write test code to verify your implementations
    dataset = Shakespeare('shakespeare_train.txt')
    print(f'Dataset length: {len(dataset)}')
    sample_input, sample_target = dataset[0]
    print(f'Sample input: {sample_input}')
    print(f'Sample target: {sample_target}')