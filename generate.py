import torch
from model import CharRNN, CharLSTM
from dataset import Shakespeare

def generate(model, seed_characters, temperature, device, char_to_idx, idx_to_char, seq_len=100):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
        temperature: T
        seq_len: length of the sequence to generate

    Returns:
        samples: generated characters
    """
    model.eval()
    input_seq = torch.tensor([char_to_idx[ch] for ch in seed_characters], dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)
    if isinstance(hidden, tuple):  # For LSTM
        hidden = (hidden[0].to(device), hidden[1].to(device))
    else:  # For RNN
        hidden = hidden.to(device)

    generated = seed_characters
    for _ in range(seq_len):
        output, hidden = model(input_seq, hidden)
        output = output / temperature
        probs = torch.nn.functional.softmax(output[-1], dim=0).detach().cpu().numpy()
        char_idx = torch.multinomial(torch.tensor(probs), 1).item()
        generated += idx_to_char[char_idx]
        input_seq = torch.tensor([[char_idx]], dtype=torch.long).to(device)

    return generated

def save_generated_text(filename, texts):
    with open(filename, 'w') as f:
        for text in texts:
            f.write(text + '\n\n')

if __name__ == '__main__':
    input_file = 'shakespeare_train.txt'
    seeds = [
        """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.""",
"""Second Citizen:
Nay, but speak not maliciously.

First Citizen:
I say unto you, what he hath done famously, he did
it to that end: though soft-conscienced men can be
content to say it was for his country he did it to
please his mother and to be partly proud; which he
is, even till the altitude of his virtue.

Second Citizen:
What he cannot help in his nature, you account a
vice in him. You must in no way say he is covetous.

First Citizen:
If I must not, I need not be barren of accusations;
he hath faults, with surplus, to tire in repetition.
What shouts are these? The other side o' the city
is risen: why stay we prating here? to the Capitol!""",
"""Second Citizen:
Worthy Menenius Agrippa; one that hath always loved
the people.

First Citizen:
He's one honest enough: would all the rest were so!

MENENIUS:
What work's, my countrymen, in hand? where go you
With bats and clubs? The matter? speak, I pray you.

First Citizen:
Our business is not unknown to the senate; they have
had inkling this fortnight what we intend to do,
which now we'll show 'em in deeds. They say poor
suitors have strong breaths: they shall know we
have strong arms too.

MENENIUS:
Why, masters, my good friends, mine honest neighbours,
Will you undo yourselves?""",
"""First Citizen:
Care for us! True, indeed! They ne'er cared for us
yet: suffer us to famish, and their store-houses
crammed with grain; make edicts for usury, to
support usurers; repeal daily any wholesome act
established against the rich, and provide more
piercing statutes daily, to chain up and restrain
the poor. If the wars eat us not up, they will; and
there's all the love they bear us.

MENENIUS:
Either you must
Confess yourselves wondrous malicious,
Or be accused of folly. I shall tell you
A pretty tale: it may be you have heard it;
But, since it serves my purpose, I will venture
To stale 't a little more.""",
"""MENENIUS:
There was a time when all the body's members
Rebell'd against the belly, thus accused it:
That only like a gulf it did remain
I' the midst o' the body, idle and unactive,
Still cupboarding the viand, never bearing
Like labour with the rest, where the other instruments
Did see and hear, devise, instruct, walk, feel,
And, mutually participate, did minister
Unto the appetite and affection common
Of the whole body. The belly answer'd--

First Citizen:
Well, sir, what answer made the belly?

MENENIUS:
Sir, I shall tell you. With a kind of smile,
Which ne'er came from the lungs, but even thus--
For, look you, I may make the belly smile
As well as speak--it tauntingly replied
To the discontented members, the mutinous parts
That envied his receipt; even so most fitly
As you malign our senators for that
They are not such as you."""
    ]
    temperature = 0.5
    seq_len = 100  # Define the length of generated sequence

    dataset = Shakespeare(input_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load RNN model
    rnn_model = CharRNN(input_size=len(dataset.chars), hidden_size=512, output_size=len(dataset.chars), n_layers=3).to(device)
    rnn_model.load_state_dict(torch.load('char_rnn.pth', map_location=device))

    # Load LSTM model
    lstm_model = CharLSTM(input_size=len(dataset.chars), hidden_size=512, output_size=len(dataset.chars), n_layers=3).to(device)
    lstm_model.load_state_dict(torch.load('char_lstm.pth', map_location=device))

    rnn_generated_texts = []
    lstm_generated_texts = []

    # Generate text using RNN model
    for seed in seeds:
        rnn_generated_text = generate(rnn_model, seed, temperature, device, dataset.char_to_idx, dataset.idx_to_char, seq_len)
        rnn_generated_texts.append(rnn_generated_text)
        print(f"RNN Generated Text:\n{rnn_generated_text}\n")

    # Generate text using LSTM model
    for seed in seeds:
        lstm_generated_text = generate(lstm_model, seed, temperature, device, dataset.char_to_idx, dataset.idx_to_char, seq_len)
        lstm_generated_texts.append(lstm_generated_text)
        print(f"LSTM Generated Text:\n{lstm_generated_text}\n")

    # Save generated texts to files
    save_generated_text('rnn_generated_texts.txt', rnn_generated_texts)
    save_generated_text('lstm_generated_texts.txt', lstm_generated_texts)
