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



for language in ["en", "nl", "fr", "es"]:

    modelfile = "../results/trained_models/morfessor/" + language + "/unsupervised.bin"
    trainfile = "../data/morfessor/wordlists/" + language + "_words.txt"
    outfile = "../results/trained_models/morfessor/" + language + "/output.csv"
    train_unsupervised(modelfile, trainfile)
