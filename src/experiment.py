import pandas as pd
import yaml
import os
import random
from datasets import load_dataset
from train_tokenization import train_tokenizer, Tokenizer, tokenize



#
# # Step 1: Train tokenizer, store model
def train(training_data,alg = "BPE", vocab_size= "10000", modeldir =""):

    trained_tokenizer = train_tokenizer([training_data], alg, vocab_size)

    modelfile = modeldir + "trained.json"
    trained_tokenizer.save(modelfile)
    with open(modeldir + "vocabulary.txt", "w") as vocfile:
        vocfile.write("\n".join(trained_tokenizer.get_vocab().keys()))
    return modelfile
#
# # Step 2: Tokenize lexical decision data
def tokenize_eval(modelfile, evaldata, outdir):
    trained_tokenizer = Tokenizer.from_file(modelfile)
    lexdata = pd.read_csv(open(evaldata, "r"), delimiter="\t")

    tokens = [str(x) for x in lexdata["spelling"]]
    lexicality = lexdata["lexicality"]
    subtokens = [(tokenize(token, trained_tokenizer)).tokens for token in tokens]
    wordiness = [1- (len(subtokens[i]) / len(tokens[i])) for i in range(len(tokens))]
    num_splits = [len(toklist)-1 for toklist in subtokens]
    reading_times = lexdata["rt"]
    lengths = [len(token) for token in tokens]
    accuracies =  lexdata["accuracy"]
    results = pd.DataFrame(list(
         zip(tokens, lexicality, lengths, subtokens, num_splits, wordiness, reading_times, accuracies            )))

    results.columns = ["Stimulus", "Lexicality", "Length", "Subtokens", "Num_Splits","Wordiness","Reading_Time", "Accuracy"]
    results = results.round(decimals=2)

    results.to_csv(outdir  + "output.csv", index = False)

    word_correlations = results[results["Lexicality"]=="W"].corr(numeric_only=True).round(decimals=2)
    nonword_correlations = results[results["Lexicality"] == "N"].corr(numeric_only=True).round(decimals=2)
    with open(outdir + "correlations.txt", "w") as corrfile:
        corrfile.write("Correlations for WORDS: \n")
        corrfile.write(word_correlations.to_string())
        corrfile.write("\n\nCorrelations for NON-WORDS: \n")
        corrfile.write(nonword_correlations.to_string())

#
# # Step 3: Evaluate
# def evaluate:
# # Step 4: Plot
print(os.getcwd())
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
    print(f)

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    print(config["vocab_size"])
    # TODO: could add some checks here to make sure config file entries are valid
    outdir = config["outdir"]
    traindir = config["traindir"]
    evaldir = config["evaldir"]
    languages = config["language"]
    algorithms = config["tokenizers"]
    vocab_sizes = config["vocab_size"]
    os.makedirs(outdir, exist_ok=True)

data = {}
for language in languages:
    print(language)

    # The parameters engine and quoting are required because the formatting of the sentences is a bit off
    training_data = pd.read_csv(traindir + language + ".txt", delimiter="\t", engine='python', names = ["Index", "Sentence"], quoting = 3)
    sentences = training_data["Sentence"].to_list()
    data[language] = sentences

    for alg in algorithms:
        print("alg")
        for vocab_size in vocab_sizes:
            print(vocab_size)

            modeldir= outdir +language +"/models/" + alg +"_"  +str(vocab_size ) + "/"
            os.makedirs(modeldir, exist_ok=True)

            print("start training")
            modelfile = train(sentences, alg,vocab_size, modeldir)
            print("done training")

            print("start tokenizing")
            evalfile = evaldir + language + ".txt"
            tokenize_eval( modelfile, evalfile ,modeldir )

# train cross-lingual
for alg in algorithms:
    print("alg")
    for vocab_size in vocab_sizes:
        print(vocab_size)

        modeldir= outdir +"crosslingual/models/" + alg +"_"  +str(vocab_size ) + "/"
        os.makedirs(modeldir, exist_ok=True)

        # Create cross-lingual dataset by concatenating and shuffleing
        all_sentences = [data[l] for l in languages]
        cross_lingual = [s for sentences in all_sentences for s in sentences]
        random.seed(5)
        random.shuffle(cross_lingual )

        print("start training")
        modelfile = train(sentences, alg,vocab_size, modeldir)
        print("done training")

        print("start tokenizing")
        for language in languages:
            evalfile = evaldir + language + ".txt"
            tokenize_eval( modelfile, evalfile ,modeldir+ language + "_" )