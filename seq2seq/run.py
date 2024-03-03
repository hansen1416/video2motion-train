import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import os
import unicodedata
import numpy as np

from Voc import Voc
from EncoderRNN import EncoderRNN
from LuongAttnDecoderRNN import LuongAttnDecoderRNN
from GreedySearchDecoder import GreedySearchDecoder

device = torch.device("cpu")


# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

MAX_LENGTH = 10  # Maximum sentence length


# Lowercase and remove non-letter characters
def normalizeString(s):
    s = s.lower()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# Takes string sentence, returns sentence of word indexes
def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(" ")] + [EOS_token]


def evaluate(searcher, voc, sentence, max_length=MAX_LENGTH):
    """
    The evaluate function takes a normalized string sentence,
    processes it to a tensor of its corresponding word indexes (with batch size of 1),
    and passes this tensor to a GreedySearchDecoder instance called searcher to handle the encoding/decoding process.

    The searcher returns the output word index vector and a scores tensor corresponding to the softmax scores for each decoded word token.
    The final step is to convert each word index back to its string representation using voc.index2word.
    """
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


# Evaluate inputs from user input (``stdin``)
def evaluateInput(searcher, voc):
    """
    The evaluateInput function prompts a user for an input, and evaluates it.
    It will continue to ask for another input until the user enters ‘q’ or ‘quit’.
    """
    input_sentence = ""
    while 1:
        try:
            # Get input sentence
            input_sentence = input("> ")
            # Check if it is quit case
            if input_sentence == "q" or input_sentence == "quit":
                break
            # Normalize sentence
            input_sentence = normalizeString(input_sentence)
            # Evaluate sentence
            output_words = evaluate(searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [
                x for x in output_words if not (x == "EOS" or x == "PAD")
            ]
            print("Bot:", " ".join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")


# Normalize input sentence and call ``evaluate()``
def evaluateExample(sentence, searcher, voc):
    """
    The evaluateExample function simply takes a string input sentence as an argument,
    normalizes it, evaluates it, and prints the response.
    """

    print("> " + sentence)
    # Normalize sentence
    input_sentence = normalizeString(sentence)
    # Evaluate sentence
    output_words = evaluate(searcher, voc, input_sentence)
    output_words[:] = [x for x in output_words if not (x == "EOS" or x == "PAD")]
    print("Bot:", " ".join(output_words))


save_dir = os.path.join("data", "save")
corpus_name = "cornell movie-dialogs corpus"

# Configure models
model_name = "cb_model"
attn_model = "dot"
# attn_model = 'general'``
# attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# If you're loading your own model
# Set checkpoint to load from
checkpoint_iter = 4000

# loadFilename = os.path.join(
#     save_dir,
#     model_name,
#     corpus_name,
#     "{}-{}_{}".format(encoder_n_layers, decoder_n_layers, hidden_size),
#     "{}_checkpoint.tar".format(checkpoint_iter),
# )

# If you're loading the hosted model
# loadFilename = os.path.join("4000_checkpoint", "4000_checkpoint.tar")
loadFilename = os.path.join("data", "4000_checkpoint.tar")

# Load model
# Force CPU device options (to match tensors in this tutorial)
checkpoint = torch.load(loadFilename, map_location=torch.device("cpu"))

# print(checkpoint)

# embedding.weight of encoder
encoder_sd = checkpoint["en"]
# embedding.weight of decoder
decoder_sd = checkpoint["de"]

encoder_optimizer_sd = checkpoint["en_opt"]
decoder_optimizer_sd = checkpoint["de_opt"]
embedding_sd = checkpoint["embedding"]


voc = Voc(corpus_name)
voc.__dict__ = checkpoint["voc_dict"]


print("Building encoder and decoder ...")
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(
    attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout
)


# Load trained model parameters
encoder.load_state_dict(encoder_sd)
decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
# Set dropout layers to ``eval`` mode
encoder.eval()
decoder.eval()
print("Models built and ready to go!")


### Compile the whole greedy search model to TorchScript model
# Create artificial inputs

# we create an example input sequence tensor test_seq,
# which is of appropriate size (MAX_LENGTH, 1),  contains numbers in the appropriate range [0,voc.num_words),
# and is of the appropriate type (int64).
test_seq = torch.LongTensor(MAX_LENGTH, 1).random_(0, voc.num_words).to(device)

# We also create a test_seq_length scalar which realistically contains the value corresponding to how many words are in the test_seq.
# The next step is to use the torch.jit.trace function to trace the model.
test_seq_length = torch.LongTensor([test_seq.size()[0]]).to(device)

# Trace the model
# Notice that the first argument we pass is the module that we want to trace,
# and the second is a tuple of arguments to the module’s forward method.
traced_encoder = torch.jit.trace(encoder, (test_seq, test_seq_length))


### Convert decoder model
# Create and generate artificial inputs
test_encoder_outputs, test_encoder_hidden = traced_encoder(test_seq, test_seq_length)
test_decoder_hidden = test_encoder_hidden[: decoder.n_layers]
test_decoder_input = torch.LongTensor(1, 1).random_(0, voc.num_words)
# Trace the model
traced_decoder = torch.jit.trace(
    decoder, (test_decoder_input, test_decoder_hidden, test_encoder_outputs)
)

### Initialize searcher module by wrapping ``torch.jit.script`` call
scripted_searcher = torch.jit.script(
    GreedySearchDecoder(traced_encoder, traced_decoder, decoder.n_layers)
)


# print("scripted_searcher graph:\n", scripted_searcher.graph)


# Use appropriate device
scripted_searcher.to(device)
# Set dropout layers to ``eval`` mode
scripted_searcher.eval()

# Evaluate examples
sentences = [
    "hello",
    "what's up?",
    "who are you?",
    "where am I?",
    "where are you from?",
]
for s in sentences:
    evaluateExample(s, scripted_searcher, voc)

# Evaluate your input by running
# evaluateInput(traced_encoder, traced_decoder, scripted_searcher, voc)
# evaluateInput(scripted_searcher, voc)


# Now that we have successfully converted our model to TorchScript,
# we will serialize it for use in a non-Python deployment environment.
# To do this, we can simply save our scripted_searcher module,
# as this is the user-facing interface for running inference against the chatbot model.
# When saving a Script module, use script_module.save(PATH) instead of torch.save(model, PATH).

# scripted_searcher.save("scripted_chatbot.pth")
