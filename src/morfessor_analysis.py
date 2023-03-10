import morfessor

import argparse
from os import makedirs, getcwd
import sys
import pandas as pd




# Set default arguments

def train_unsupervised(modelfile, trainfile):
    parser = morfessor.get_default_argparser()
    parser.prog = "morfessor-train"

    keep_options = ['savesegfile', 'savefile', 'trainmode', 'dampening',
                        'encoding', 'list', 'skips', 'annofile', 'develfile',
                        'fullretrain', 'threshold', 'morphtypes', 'morphlength',
                        'corpusweight', 'annotationweight', 'help', 'version']
    for action_group in parser._action_groups:
        for arg in action_group._group_actions:
            if arg.dest not in keep_options:
                print(arg)
                arg.help = argparse.SUPPRESS
    parser.add_argument('trainfiles', metavar='<file>', nargs='+',
                            help='training data files')
    # Training

    makedirs(modelfile, exist_ok=True)
    # I am running the model with the default unsupervised settings and the Leipzig training data
    arguments = ["-s", modelfile, "--traindata-list", trainfile]

    try:

        args = parser.parse_args(arguments)

        morfessor.main(args)
    except morfessor.ArgumentException as e:
        parser.error(e)
    except Exception as e:
        morfessor._logger.error("Fatal Error {} {}".format(type(e), e))
        raise

# Evaluation
# TODO: this is almost the same code as for evaluating the subtokenizers, could be refactored
def evaluate(modelfile, evaluationfile, outfile):
    model = morfessor.io.MorfessorIO().read_binary_model_file(modelfile)
    lexdata = pd.read_csv(evaluationfile, sep="\t")

    tokens = [str(x) for x in lexdata["spelling"]]
    lexicality = lexdata["lexicality"]
    morfessor_output = [model.viterbi_segment(token) for token in tokens]
    morphemes = [x[0] for x in morfessor_output]
    scores = [x[1] for x in morfessor_output]

    wordiness = [1-(len(morphemes[i]) / len(tokens[i])) for i in range(len(tokens))]
    lengths = [len(token) for token in tokens]

    num_splits = [len(toklist) - 1 for toklist in morphemes]
    reading_times = lexdata["rt"]
    accuracies = lexdata["accuracy"]
    results = pd.DataFrame(list(
        zip(tokens, lexicality, lengths, morphemes, num_splits, wordiness, scores, reading_times, accuracies)))

    results.columns = ["Stimulus", "Lexicality", "Length", "Subtokens", "Num_Splits", "Wordiness", "Morphessor_Scores", "Reading_Time",
                       "Accuracy"]
    results = results.round(decimals=2)

    results.to_csv(outfile, index=False)

    words = results[results["Lexicality"]=="W"]
    nonwords = results[results["Lexicality"]=="N"]
    print(words.corr().round(decimals=2).to_string())
    print(nonwords.corr().round(decimals=2).to_string())

for language in ["en", "nl", "fr", "es"]:

    modelfile = "../morfessor/models/" + language + "/unsupervised.bin"
    trainfile = "../data/morfessor/wordlists/" + language + "_words.txt"
    evalfile = "../data/eval/" + language + ".txt"
    outfile = "../morfessor/models/" + language + "/output.csv"
    #train_unsupervised(modelfile, trainfile)
    evaluate(modelfile, evalfile, outfile)