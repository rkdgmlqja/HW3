import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from dataset import Shakespeare
from model import CharRNN, CharLSTM

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
    """

    # write your codes here
    model.train()
    trn_loss = 0

    for inputs, targets in trn_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        hidden = model.init_hidden(batch_size)
        if isinstance(hidden, tuple):  # For LSTM
            hidden = (hidden[0].to(device), hidden[1].to(device))
        else:  # For RNN
            hidden = hidden.to(device)
        optimizer.zero_grad()

        output, hidden = model(inputs, hidden)
        loss = criterion(output, targets.view(-1))
        loss.backward()
        optimizer.step()
        trn_loss += loss.item()

    return trn_loss / len(trn_loader)

def validate(model, val_loader, device, criterion):
    """ Validate function

    Args:
        model: network
        val_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        val_loss: average loss value
    """

    # write your codes here
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            hidden = model.init_hidden(batch_size)
            if isinstance(hidden, tuple):  # For LSTM
                hidden = (hidden[0].to(device), hidden[1].to(device))
            else:  # For RNN
                hidden = hidden.to(device)

            output, hidden = model(inputs, hidden)
            loss = criterion(output, targets.view(-1))
            val_loss += loss.item()

    return val_loss / len(val_loader)

def main():
    """ Main function

        Here, you should instantiate
        1) DataLoaders for training and validation.
           Try SubsetRandomSampler to create these DataLoaders.
        3) model
        4) optimizer
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    # write your codes here
    input_file = 'shakespeare_train.txt'
    batch_size = 1024
    epochs = 100
    hidden_size = 512
    n_layers = 3
    learning_rate = 0.003

    dataset = Shakespeare(input_file)
    val_split = int(0.1 * len(dataset))
    train_split = len(dataset) - val_split
    train_dataset, val_dataset = random_split(dataset, [train_split, val_split])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    char_rnn = CharRNN(input_size=len(dataset.chars), hidden_size=hidden_size, output_size=len(dataset.chars), n_layers=n_layers).to(device)
    char_lstm = CharLSTM(input_size=len(dataset.chars), hidden_size=hidden_size, output_size=len(dataset.chars), n_layers=n_layers).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_rnn = optim.Adam(char_rnn.parameters(), lr=learning_rate)
    optimizer_lstm = optim.Adam(char_lstm.parameters(), lr=learning_rate)

    scheduler_rnn = optim.lr_scheduler.ReduceLROnPlateau(optimizer_rnn, mode='min', factor=0.5, patience=5,
                                                         verbose=True)
    scheduler_lstm = optim.lr_scheduler.ReduceLROnPlateau(optimizer_lstm, mode='min', factor=0.5, patience=5,
                                                          verbose=True)

    best_val_loss_rnn = float('inf')
    best_val_loss_lstm = float('inf')

    for epoch in range(epochs):
        train_loss_rnn = train(char_rnn, train_loader, device, criterion, optimizer_rnn)
        val_loss_rnn = validate(char_rnn, val_loader, device, criterion)

        if val_loss_rnn < best_val_loss_rnn:
            best_val_loss_rnn = val_loss_rnn
            torch.save(char_rnn.state_dict(), 'char_rnn.pth')
        scheduler_rnn.step(val_loss_rnn)

        train_loss_lstm = train(char_lstm, train_loader, device, criterion, optimizer_lstm)
        val_loss_lstm = validate(char_lstm, val_loader, device, criterion)

        if val_loss_lstm < best_val_loss_lstm:
            best_val_loss_lstm = val_loss_lstm
            torch.save(char_lstm.state_dict(), 'char_lstm.pth')
        scheduler_lstm.step(val_loss_lstm)

        print(f'Epoch {epoch+1}/{epochs} - RNN Train Loss: {train_loss_rnn:.4f} Val Loss: {val_loss_rnn:.4f}')
        print(f'Epoch {epoch+1}/{epochs} - LSTM Train Loss: {train_loss_lstm:.4f} Val Loss: {val_loss_lstm:.4f}')

if __name__ == '__main__':
    main()