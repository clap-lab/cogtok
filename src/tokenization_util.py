## importing the tokenizer and subword BPE trainer
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordLevel, WordPiece
from tokenizers.trainers import BpeTrainer, WordLevelTrainer, \
    WordPieceTrainer, UnigramTrainer
## a pretokenizer to segment the text into words
from tokenizers.pre_tokenizers import Whitespace


unk_token = "<UNK>"  # token for unknown words
spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]  # special tokens


def prepare_tokenizer_trainer(alg, vocab_size):
    """
    Prepares the tokenizer and trainer with unknown & special tokens.
    """

    if alg == 'BPE':
        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        trainer = BpeTrainer(special_tokens=spl_tokens, vocab_size=vocab_size)
    elif alg == 'UNI':
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(unk_token=unk_token, special_tokens=spl_tokens, vocab_size=vocab_size)
    elif alg == 'WPC':
        tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
        trainer = WordPieceTrainer(special_tokens=spl_tokens, vocab_size=vocab_size)
    else:
        tokenizer = Tokenizer(WordLevel(unk_token=unk_token))
        trainer = WordLevelTrainer(special_tokens=spl_tokens, vocab_size=vocab_size)

    tokenizer.pre_tokenizer = Whitespace()

    return tokenizer, trainer


def train_tokenizer(data, alg='WLV', vocab_size = 10000):
    """
    Takes the files and trains the tokenizer.
    """
    tokenizer, trainer = prepare_tokenizer_trainer(alg, vocab_size)

    tokenizer.train_from_iterator(data, trainer)  # training the tokenzier

    return tokenizer


def tokenize(input_string, tokenizer):
    """
    Tokenizes the input string using the tokenizer provided.
    """
    output = tokenizer.encode(input_string)
    return output

