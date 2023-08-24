import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

evalpath = "../../data/eval/"
resultpath = "../../results/"
outdir = resultpath + "results_overview/"
langs = ["en", "nl", "fr", "es"]
model = "Model_WPC_50000"
signal = "accuracy"

def process_dataset(dataset, signal, model):
    measure = dataset[signal].astype(float).to_numpy()

    # Alternative: Trying out a length baseline
    # stimuli = dataset["spelling"].to_numpy()
    # measure = np.asarray([len(x) for x in stimuli ])

    measure = measure.reshape(-1, 1)

    # focus on specific model
    splits = list(literal_eval(tokens) for tokens in dataset[model])
    feature = np.asarray([1 - (len(tokens) / len("".join(tokens))) for tokens in splits])
    feature = feature.reshape(-1, 1)

    return feature, measure

def map_labels(low_threshold, high_threshold, features, labels):
    # print("Original size:")
    # print(len(features), len(labels))
    reduced_features = []
    reduced_labels = []
    for i, l in enumerate(labels):
        if l<low_threshold:
            reduced_features.append(features[i])
            reduced_labels.append("low")
        if l > high_threshold:
            reduced_features.append(features[i])
            reduced_labels.append("high")

    # print("Reduced size:")
    # print(len(reduced_features), len(reduced_labels))
    return reduced_features, reduced_labels

for lang in langs:
    print("\n\n")
    print(lang)

    evaldata = pd.read_csv(resultpath + "results_overview/" + lang + ".csv")
    evaldata = evaldata.dropna()

    words = evaldata[evaldata["lexicality"]=="W"]
    nonwords = evaldata[evaldata["lexicality"]== "N"]

    datasets = [words, nonwords]
    all_results = {}

    # Switch between words and nonwords
    for category in ["words", "nonwords"]:
        print(category)
        if category == "words":
            dataset = datasets[0]
        else:
            dataset = datasets[1]

        # Prepare train and test
        seed = 1
        train, test = train_test_split(dataset, test_size=0.2)
        # using reading time here, alternative would be Accuracy
        train_features, train_signal = process_dataset(train, signal, model)
        test_features, test_signal = process_dataset(test, signal, model)

        # Determine thresholds on train set
        low_threshold = np.percentile(train_signal, 25)
        high_threshold = np.percentile(train_signal, 75)

        # Map labels into high (>high_threshold) and low(<low_threshold) and discard other instances
        train_features, train_labels = map_labels(low_threshold, high_threshold, train_features, train_signal)
        test_features, test_labels = map_labels(low_threshold, high_threshold, test_features, test_signal)

        # Classify
        classifiers = [GaussianNB(), SVC(kernel="linear", C=0.025), KNeighborsClassifier(3)]
        for classifier in classifiers:
            classifier.fit(train_features, train_labels)
            score = classifier.score(test_features, test_labels)
            print(score)

