import random

import pandas as pd
from tokenization_util import train_tokenizer
from morphology_util import evaluate_derivational_coverage

sizes = [100000]
vocab_sizes = [500, 1000, 2500, 5000, 7500, 10000, 20000, 30000, 40000, 50000]

languages = ["hun", "por", "cat", "fin", "ita", "rus", "ces", "fra", "eng", "mon", "spa", "deu", "hbs", "pol", "swe"]
# vary size of training data

algs = ["BPE", "WPC", "UNI"]
traindir = "train/"
seed = 42
derivational_characteristics = {}


# Calculating
for language in languages:
    print(language)
    results = []
    training_data = pd.read_csv(traindir + language + ".txt", delimiter="\t", engine='python', names=["Index", "Sentence"], quoting=3)
    sentences = training_data["Sentence"].to_list()
    for alg in algs:
        for training_size in sizes:
            for vocab_size in vocab_sizes:
                # 1. Train tokenizer

                print("Training: ")
                print(training_size, vocab_size)
                data = random.sample(sentences, training_size)
                trained_tokenizer = train_tokenizer([data], alg, vocab_size)
                vocabulary = trained_tokenizer.get_vocab().keys()
                modeldir = "../results/derivational_overlap/trained_models/"+ language + "_" + alg + "_" + str(vocab_size) + "_"
                trained_tokenizer.save(modeldir + "trained.json")
                with open(modeldir + "vocabulary.txt", "w", encoding="utf-8") as vocfile:
                    vocfile.write("\n".join(vocabulary))

                # 2. Calculate derivational overlap
                covered, overlap, threshold, morphemes = evaluate_derivational_coverage(language, vocabulary)
                if language not in derivational_characteristics:
                    derivational_characteristics[language] = (threshold, len(morphemes))
                overlap = round(overlap, 2)
                print("Done Training. Overlap: " + str(overlap))
                print("\n\n")
                results.append([language, alg, training_size, vocab_size, overlap])

    print(results)
    results_frame = pd.DataFrame(results, columns = ["language", "algorithm", "training_size", "vocab_size", "overlap"])
    results_frame.to_csv("../results/derivational_overlap/" + language + ".csv")
