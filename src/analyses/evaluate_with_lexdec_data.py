import pandas as pd
import yaml
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
import os
from ast import literal_eval
from scipy.stats import pearsonr

evalpath = "../data/eval/"
resultpath = "../results/"
outdir = resultpath + "results_overview/"

# Collect all outputs of all trained trained_models in a single file
def collect_all_outputs(langs):
    for lang in langs:
        print(lang)

        # Note: we read in the eval data again because I made a change in the pre-processing of Spanish and the columns for reading time and accuracy in output.csv are slightly outdated [for Spanish]. If would have been cleaner not to add them to the output file from the start
        evaldata = pd.read_csv(open(evalpath + lang + ".txt", "r"), delimiter="\t")
        modelpath = resultpath  + lang +"/trained_models/"
        models = []
        outputs = {key: [] for key in evaldata["spelling"]}

        for model in [f for f in os.listdir(modelpath) if not f.startswith('.')]:
            print(model)
            models.append(model)
            outputpath = modelpath + model + "/output.csv"

            # extract splits from model output
            modeloutput = pd.read_csv(outputpath, delimiter=",", index_col=False)
            segmentations = dict(zip(modeloutput["Stimulus"], modeloutput["Subtokens"]))

            # Sanity check:     assert(len(outputs)== len(modeloutput["Stimulus"]))
            for key in outputs:
                previous_outputs = outputs[key]

                if segmentations[key]:
                    previous_outputs.append(segmentations[key])
                else:
                    print("No model output for: " + key)
                    previous_outputs.append([])
                outputs[key] = previous_outputs

    # This is a bit of a duplication but collecting the data in pandas directly costs too much memory
        columns = ["spelling"] + ["Model_" + x for x in models]
        print(columns)
        outputdata = pd.DataFrame(columns=columns)
        outputdata["spelling"] = outputs.keys()
        for i, model in enumerate(models):
            outputdata["Model_" + model] = [item[i] for item in outputs.values()]

        # Save a copy in case merge goes wrong
        outputdata.to_csv(outdir  + lang + "_onlyoutput.csv", index = False)
        evaldata = evaldata.merge(outputdata, how="left", on="spelling")
        evaldata.to_csv(outdir  + lang + ".csv", index = False)

langs = ["en", "nl", "fr", "es"]
collect_all_outputs(langs)

# Evaluate with lexical decision data
for lang in langs:

    evaldata = pd.read_csv(resultpath + "results_overview/" + lang + ".csv")
    evaldata = evaldata.dropna()
    words = evaldata[evaldata["lexicality"]=="W"]
    nonwords = evaldata[evaldata["lexicality"]== "N"]

    datasets = [words, nonwords]
    all_results = {}
    for category in ["words", "nonwords"]:
        print(category)
        if category == "words":
            dataset = datasets[0]
        else:
            dataset = datasets[1]

        correlation_with_numsplits = []
        correlation_with_wordiness = []

        category_results = []
        for model in dataset:

            if model.startswith("Model"):


                # measurements
                tokens = list(dataset["spelling"])
                rts = list(dataset["rt"])
                accs = list(dataset["accuracy"])

                # splits in model output
                splits = list(dataset[model].apply(literal_eval))
                num_splits = [len(x)-1 for x in splits]
                max = len(dataset[model])
                wordiness = [1-(len(splits[i])/len(str(tokens[i]))) for i in range(max)]

                # correlation
                corr1 = pearsonr(num_splits, rts)[0]
                corr2 = pearsonr(num_splits, accs)[0]
                corr3 = pearsonr(wordiness, rts)[0]
                corr4 = pearsonr(wordiness, accs)[0]

                results = [model, "{:.2f}".format(corr1) , "{:.2f}".format(corr2),"{:.2f}".format(corr3), "{:.2f}".format(corr4)]
                category_results.append(results)
                print(results)
        all_results[category] = category_results

    with open(outdir +  lang + "_correlations.csv", "w") as outfile:
        for category in all_results.keys():
            outfile.write(category)
            outfile.write("\n")
            outfile.write("Model, Vocab_Size, NumSplits_RT, NumSplits_Acc, Chunkability_RT, Chunkability_Acc\n")
            for line in all_results[category]:
                modelname, result1, result2, result3, result4 = line
                try:
                    _, name, size = modelname.split("_")
                except ValueError:
                    _, name = modelname.split("_")
                    size = "default"

                outfile.write(",".join([name, size, result1, result2, result3, result4]))
                outfile.write("\n")

