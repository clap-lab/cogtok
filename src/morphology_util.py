import pandas as pd
import warnings
from string import punctuation
from tokenization_util import tokenize, Tokenizer


def read_derivational(language):

    if len(language) != 3:
        raise ValueError("Morphynet uses three-letter language codes")

    # Set path
    datadir = "../data/morphynet2/" + language + "/"
    derivational_file = datadir + language + ".derivational.v1.tsv"

    # Set header
    derivational_header = ["source_word", "target_word", "source_POS", "target_POS", "morpheme", "type"]

    # Read file
    derivational = pd.read_csv(derivational_file, sep="\t", names=derivational_header, keep_default_na=False)

    return derivational


# This is not used
def read_inflectional(language):
    if len(language) != 3:
        raise ValueError("Morphynet uses three-letter language codes")

    # Set path
    datadir = "../data/morphynet/" + language + "/"
    inflectional_file = datadir + language + ".inflectional.v1.tsv"

    # Set header
    inflectional_header = ["lemma", "inflected_form", "morphological_features", "morpheme_segmentation"]

    # Read files
    inflectional = pd.read_csv(inflectional_file, sep="\t", names=inflectional_header, keep_default_na=False)
    return inflectional


def evaluate_derivational_coverage(language, vocab):
    derivational = read_derivational(language)
    counts = derivational["morpheme"].value_counts()
    threshold = int(len(derivational) * 0.001)
    derivational_morphemes = set(derivational[derivational["morpheme"].map(counts)>threshold]["morpheme"])
    # print(threshold)
    # print("Number of derivational morphemes: " + str(len(derivational_morphemes)))
    # print(derivational_morphemes)

    covered = []
    for token in derivational_morphemes:
        if token in vocab or token.lower() in vocab or ("##" + token) in vocab:
            covered.append(token)
    coverage = len(covered) / len(derivational_morphemes)

    # print("Derivational morphemes covered in vocab: {:.2f}".format(coverage))
    # print(len(covered), len(derivational_morphemes))
    # TODO: need to double-check whether they occur in training data at all

    return covered, coverage, threshold, derivational_morphemes

def get_derivation(sequence, derivational_segmentation):

    if not sequence in derivational_segmentation.keys():
        return [sequence]
    else:
        morphemes = []
        derivative = derivational_segmentation[sequence]
        before, derivative, after = sequence.partition(derivative)

        # Recursion step
        if not len(derivative) == 0:
            if len(before)> 0:
                before_splits = get_derivation(before, derivational_segmentation)
                morphemes.extend(before_splits)

            morphemes.extend([derivative])

            if len(after) > 0:
                after_splits = get_derivation(after, derivational_segmentation)
                morphemes.extend(after_splits)

            return morphemes

        # This captures two cases
        # 1. Faulty entries in morphynet, e.g. the derivative of elope is marked as "and"
        # 2. Phonetic changes, e.g. the derivative of discernible is "able".
        # TODO: It would be good to come up with a solution for phonetic changes.
        else:
            warning = "Ignoring irregular entry in morphynet: " + str(sequence) + ": " +  str(derivational_segmentation[sequence])
            warnings.warn(warning)
            return [sequence]

# This function splits a sequence into the morphemes according to the data in the segmentation dictionaries (we use morphynet).
# If there is no entry, the sequence is not split.
# The challenge is that we need to combine inflectional and derivational splits
# I tried to check the functionality carefully for English, ... but it might still contain errors.

def split(sequence, inflectional_segmentation, derivational_segmentation):
        try:
            # TODO: inflectional segmentation is not exact, e.g. trophies becomes trophy|s
            inflectional_morphemes = inflectional_segmentation[sequence].split("|")

            if len(inflectional_morphemes) == 1:
                raise KeyError

        except KeyError:
            inflectional_morphemes = [sequence]
        # TODO need to account for phonotactic changes
        #if not "".join(inflectional_morphemes) == sequence:
            #print("phonetic change", sequence, inflectional_morphemes)

        morphemes = []
        for subsequence in inflectional_morphemes:
            morphemes.extend(get_derivation(subsequence, derivational_segmentation))
        return morphemes

def evaluate_segmentation(language, trained_tokenizer):
    derivational = read_derivational(language)
    inflectional = read_inflectional(language)

    inflectional_segmentation = dict(zip(inflectional["inflected_form"], inflectional["morpheme_segmentation"]))
    derivational_segmentation = dict(zip(derivational["target_word"], derivational["morpheme"]))

    # Morphynet contains some weird entries
    matched = []
    split_match = []
    segmentation ={}
    for sequence in inflectional_segmentation.keys():
        # CHeck for punctuation and whitespaces
        if not any(p in sequence for p in punctuation) and len(sequence.split(" "))==1:
            morphemes = split(sequence, inflectional_segmentation, derivational_segmentation)
            subtokens = tokenize(sequence, trained_tokenizer).tokens
            matched.append(morphemes == subtokens)
            split_match.append(len(morphemes)==len(subtokens))
            segmentation[sequence] = (morphemes, subtokens)
    print(sum(matched), len(matched))
    print("Matched: {:.2f}".format(sum(matched)/len(matched)))
    print("Same number of splits: {:.2f}".format(sum(split_match) / len(matched)))
    return segmentation




