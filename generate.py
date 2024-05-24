# import some packages you need here
import torch
from model import CharRNN, CharLSTM
from dataset import Shakespeare

# def generate(model, seed_characters, temperature, *args):
#     """ Generate characters
#
#     Args:
#         model: trained model
#         seed_characters: seed characters
# 				temperature: T
# 				args: other arguments if needed
#
#     Returns:
#         samples: generated characters
#     """
#     model.eval()
#     input_seq = torch.tensor([char_to_idx[ch] for ch in seed_characters], dtype=torch.long).unsqueeze(0).to(device)
#     hidden = model.init_hidden(1)
#
#     generated = seed_characters
#     # write your codes here
#     for _ in range(seq_len):
#         output, hidden = model(input_seq, hidden)
#         output = output / temperature
#         probs = torch.nn.functional.softmax(output[-1], dim=0).cpu().numpy()
#         char_idx = torch.multinomial(torch.tensor(probs), 1).item()
#         generated += idx_to_char[char_idx]
#         input_seq = torch.tensor([[char_idx]], dtype=torch.long).to(device)
#
#     return samples

import torch
from model import CharRNN, CharLSTM
from dataset import Shakespeare

def generate(model, seed_characters, temperature, device, char_to_idx, idx_to_char, seq_len=100):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
				temperature: T
				args: other arguments if needed

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

if __name__ == '__main__':
    input_file = 'shakespeare_train.txt'
    model_type = 'rnn'  # Choose 'rnn' or 'lstm'

    seed1 = """MENENIUS:
    Good my friends,
    If you have heard your general talk of Rome,
    And of his friends there, it is lots to blanks,
    My name hath touch'd your ears it is Menenius.
    
    First Senator:
    Be it so; go back: the virtue of your name
    Is not here passable.
    
    MENENIUS:
    I tell thee, fellow,
    The general is my lover: I have been
    The book of his good acts, whence men have read
    His name unparallel'd, haply amplified;
    For I have ever verified my friends,
    Of whom he's chief, with all the size that verity
    Would without lapsing suffer: nay, sometimes,
    Like to a bowl upon a subtle ground,
    I have tumbled past the throw; and in his praise
    Have almost stamp'd the leasing: therefore, fellow,
    I must have leave to pass.
    
    First Senator:
    Faith, sir, if you had told as many lies in his
    behalf as you have uttered words in your own, you
    should not pass here; no, though it were as virtuous
    to lie as to live chastely. Therefore, go back.
    
    MENENIUS:
    Prithee, fellow, remember my name is Menenius,
    always factionary on the party of your general."""

    seed2 = """GLOUCESTER:
    Go you before, and I will follow you.
    He cannot live, I hope; and must not die
    Till George be pack'd with post-horse up to heaven.
    I'll in, to urge his hatred more to Clarence,
    With lies well steel'd with weighty arguments;
    And, if I fall not in my deep intent,
    Clarence hath not another day to live:
    Which done, God take King Edward to his mercy,
    And leave the world for me to bustle in!
    For then I'll marry Warwick's youngest daughter.
    What though I kill'd her husband and her father?
    The readiest way to make the wench amends
    Is to become her husband and her father:
    The which will I; not all so much for love
    As for another secret close intent,
    By marrying her which I must reach unto.
    But yet I run before my horse to market:
    Clarence still breathes; Edward still lives and reigns:
    When they are gone, then must I count my gains.

    LADY ANNE:
    Set down, set down your honourable load,
    If honour may be shrouded in a hearse,
    Whilst I awhile obsequiously lament
    The untimely fall of virtuous Lancaster.
    Poor key-cold figure of a holy king!
    Pale ashes of the house of Lancaster!
    Thou bloodless remnant of that royal blood!
    Be it lawful that I invocate thy ghost,
    To hear the lamentations of Poor Anne,
    Wife to thy Edward, to thy slaughter'd son,
    Stabb'd by the selfsame hand that made these wounds!
    Lo, in these windows that let forth thy life,
    I pour the helpless balm of my poor eyes.
    Cursed be the hand that made these fatal holes!
    Cursed be the heart that had the heart to do it!
    Cursed the blood that let this blood from hence!
    More direful hap betide that hated wretch,
    That makes us wretched by the death of thee,
    Than I can wish to adders, spiders, toads,
    Or any creeping venom'd thing that lives!
    If ever he have child, abortive be it,
    Prodigious, and untimely brought to light,
    Whose ugly and unnatural aspect
    May fright the hopeful mother at the view;
    And that be heir to his unhappiness!
    If ever he have wife, let her he made
    A miserable by the death of him
    As I am made by my poor lord and thee!
    Come, now towards Chertsey with your holy load,
    Taken from Paul's to be interred there;
    And still, as you are weary of the weight,
    Rest you, whiles I lament King Henry's corse."""

    seed3 = """SICINIUS:
    He's sentenced; no more hearing.
    
    COMINIUS:
    Let me speak:
    I have been consul, and can show for Rome
    Her enemies' marks upon me. I do love
    My country's good with a respect more tender,
    More holy and profound, than mine own life,
    My dear wife's estimate, her womb's increase,
    And treasure of my loins; then if I would
    Speak that,--
    
    SICINIUS:
    We know your drift: speak what?
    
    BRUTUS:
    There's no more to be said, but he is banish'd,
    As enemy to the people and his country:
    It shall be so.
    
    Citizens:
    It shall be so, it shall be so.
    
    CORIOLANUS:
    You common cry of curs! whose breath I hate
    As reek o' the rotten fens, whose loves I prize
    As the dead carcasses of unburied men
    That do corrupt my air, I banish you;
    And here remain with your uncertainty!
    Let every feeble rumour shake your hearts!
    Your enemies, with nodding of their plumes,
    Fan you into despair! Have the power still
    To banish your defenders; till at length
    Your ignorance, which finds not till it feels,
    Making not reservation of yourselves,
    Still your own foes, deliver you as most
    Abated captives to some nation
    That won you without blows! Despising,
    For you, the city, thus I turn my back:
    There is a world elsewhere.
    
    AEdile:
    The people's enemy is gone, is gone!"""

    seed4="""CORIOLANUS:
    What is the matter
    That being pass'd for consul with full voice,
    I am so dishonour'd that the very hour
    You take it off again?
    
    SICINIUS:
    Answer to us.
    
    CORIOLANUS:
    Say, then: 'tis true, I ought so.
    
    SICINIUS:
    We charge you, that you have contrived to take
    From Rome all season'd office and to wind
    Yourself into a power tyrannical;
    For which you are a traitor to the people.
    
    CORIOLANUS:
    How! traitor!
    
    MENENIUS:
    Nay, temperately; your promise.
    
    CORIOLANUS:
    The fires i' the lowest hell fold-in the people!
    Call me their traitor! Thou injurious tribune!
    Within thine eyes sat twenty thousand deaths,
    In thy hand clutch'd as many millions, in
    Thy lying tongue both numbers, I would say
    'Thou liest' unto thee with a voice as free
    As I do pray the gods.
    
    SICINIUS:
    Mark you this, people?
    
    Citizens:
    To the rock, to the rock with him!
    
    SICINIUS:
    Peace!
    We need not put new matter to his charge:
    What you have seen him do and heard him speak,
    Beating your officers, cursing yourselves,
    Opposing laws with strokes and here defying
    Those whose great power must try him; even this,
    So criminal and in such capital kind,
    Deserves the extremest death."""

    seed5 = """AEdile:
    With old Menenius, and those senators
    That always favour'd him.
    
    SICINIUS:
    Have you a catalogue
    Of all the voices that we have procured
    Set down by the poll?
    
    AEdile:
    I have; 'tis ready.
    
    SICINIUS:
    Have you collected them by tribes?
    
    AEdile:
    I have.
    
    SICINIUS:
    Assemble presently the people hither;
    And when they bear me say 'It shall be so
    I' the right and strength o' the commons,' be it either
    For death, for fine, or banishment, then let them
    If I say fine, cry 'Fine;' if death, cry 'Death.'
    Insisting on the old prerogative
    And power i' the truth o' the cause.
    
    AEdile:
    I shall inform them.
    
    BRUTUS:
    And when such time they have begun to cry,
    Let them not cease, but with a din confused
    Enforce the present execution
    Of what we chance to sentence.
    
    AEdile:
    Very well.
    
    SICINIUS:
    Make them be strong and ready for this hint,
    When we shall hap to give 't them.
    
    BRUTUS:
    Go about it.
    Put him to choler straight: he hath been used
    Ever to conquer, and to have his worth
    Of contradiction: being once chafed, he cannot
    Be rein'd again to temperance; then he speaks
    What's in his heart; and that is there which looks
    With us to break his neck.
    
    SICINIUS:
    Well, here he comes.
    
    MENENIUS:
    Calmly, I do beseech you.
    """
    temperature = 0.5

    dataset = Shakespeare(input_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == 'rnn':
        model = CharRNN(input_size=len(dataset.chars), hidden_size=512, output_size=len(dataset.chars), n_layers=3).to(device)
        model.load_state_dict(torch.load('char_rnn.pth', map_location=device))
    else:
        model = CharLSTM(input_size=len(dataset.chars), hidden_size=512, output_size=len(dataset.chars), n_layers=3).to(device)
        model.load_state_dict(torch.load('char_lstm.pth', map_location=device))

    generated_text = generate(model, seed, temperature, device, dataset.char_to_idx, dataset.idx_to_char)
    print(generated_text)