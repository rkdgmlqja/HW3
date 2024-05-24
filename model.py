import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        # write your codes here
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # write your codes here
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output.contiguous().view(output.size(0) * output.size(1), -1))
        return output, hidden

    def init_hidden(self, batch_size):
        # write your codes here
        return torch.zeros(self.n_layers, batch_size, self.hidden_size)

class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(CharLSTM, self).__init__()
        # write your codes here
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # write your codes here
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output.contiguous().view(output.size(0) * output.size(1), -1))
        return output, hidden

    def init_hidden(self, batch_size):
        # write your codes here
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size),
                torch.zeros(self.n_layers, batch_size, self.hidden_size))