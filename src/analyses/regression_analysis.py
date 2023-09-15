import pandas as pd
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt
from sklearn import linear_model, preprocessing
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split

from wordfreq import word_frequency, zipf_frequency


evalpath = "../../data/eval/"
resultpath = "../../results/"
outdir = resultpath + "results_overview/"
langs = ["en", "nl", "fr", "es"]
tokenizer = "Model_WPC_50000"
signal = "accuracy"


def process_dataset(dataset, signal, model):
    measure = dataset[signal].astype(float).to_numpy()

    # Alternative: Trying out a length baseline
    # stimuli = dataset["spelling"].to_numpy()
    # measure = np.asarray([len(x) for x in stimuli ])

    measure = measure.reshape(-1, 1)

    # focus on specific model

    splits = []
    for tokens in dataset[model]:
        try:
            splits.append(list(literal_eval(tokens)))
        except KeyError:
            print(tokens)
    feature = np.asarray([1 - (len(tokens) / len("".join(tokens))) for tokens in splits])
    feature = feature.reshape(-1, 1)

    return feature, measure

def calculate_frequency(dataset, language):
    frequencies = []
    for word in dataset["spelling"]:
        frequencies.append(zipf_frequency(word, language))
    frequencies = np.asarray(frequencies).reshape(-1, 1)
    print(frequencies[:50])
    return frequencies

for lang in langs:

    print("\n\n-------------\n")
    print(lang)

    evaldata = pd.read_csv(resultpath + "results_overview/" + lang + ".csv")
    evaldata = evaldata.dropna()

    words = evaldata[evaldata["lexicality"] == "W"]

    # Prepare train and test
    seed = 1
    #train =words
    train, test = train_test_split(words, test_size=0.2)

    for signal in ["rt", "accuracy"]:
        print()
        print(signal)
        # using reading time here, alternative would be Accuracy
        train_chunkability, train_signal = process_dataset(train, signal, tokenizer)
        test_chunkability, test_signal = process_dataset(test, signal, tokenizer)

        #normalize signal
        min_max_scaler = preprocessing.MinMaxScaler()
        train_signal = min_max_scaler.fit_transform(train_signal)
        test_signal = min_max_scaler.fit_transform(test_signal)
        # calculate frequency
        train_frequencies = calculate_frequency(train, lang)
        test_frequencies = calculate_frequency(test, lang)

        model = linear_model.LinearRegression()

        # fit on chunkability
        model.fit(train_chunkability, train_signal)
        training_predictions = model.predict(train_chunkability)
        print("\nFit with chunkability on train")
        print("Mean squared error: %.2f" % mean_squared_error(training_predictions, train_signal))
        print("Explained Variance: %.2f" % explained_variance_score(training_predictions, train_signal))


        test_predictions = model.predict(test_chunkability)
        print("Fit with chunkability on test")
        print("Mean squared error: %.2f" % mean_squared_error(test_predictions, test_signal))
        print("Explained Variance: %.2f" % explained_variance_score(test_predictions, test_signal))


        # fit on frequencies
        model.fit(train_frequencies, train_signal)
        training_predictions = model.predict(train_frequencies)
        print("\nFit with frequencies on train")
        print("Mean squared error: %.2f" % mean_squared_error(training_predictions, train_signal))
        print("Explained Variance: %.2f" % explained_variance_score(training_predictions, train_signal))

        test_predictions = model.predict(test_frequencies)
        print("Fit with frequency on test")
        print("Mean squared error: %.2f" % mean_squared_error(test_predictions, test_signal))
        print("Explained Variance: %.2f" % explained_variance_score(test_predictions, test_signal))

