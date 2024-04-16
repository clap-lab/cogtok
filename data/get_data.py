import requests, zipfile, io
import pandas as pd
import tarfile
import os
import shutil
from bs4 import BeautifulSoup as bs


def preprocess_fr_eval_data():
    print("Pre-processing French lexical decision data")
    nonwords = pd.read_excel("eval/raw/Ferrand-BRM-2010/FLP-pseudowords.xls", index_col=None, header=0)
    nonwords.reset_index(drop=True, inplace=True)
    words = pd.read_excel("eval/raw/Ferrand-BRM-2010/FLP-words.xls", index_col=None, header=0, usecols=range(7))
    words.reset_index(drop=True, inplace=True)
    words["lexicality"] = ["W" for x in range(len(words))]
    nonwords["lexicality"] = ["N" for x in range(len(nonwords))]
    # concat the two frames, add lexicality value based on source dataset
    french_data = pd.concat([nonwords, words], axis=0)

    # Columns: item, n_trials, err, rt, sd, rtz, n_used
    # Accuracy is 1-err
    french_data["accuracy"] = (1 - french_data["err"]).round(decimals=2)
    french_data["rt"] = french_data["rt"].round(decimals=2)
    french_data = french_data.rename(columns={"item": "spelling"})
    mapped_data = french_data[["spelling", "lexicality", "rt", "accuracy"]]
    mapped_data.to_csv("eval/fr.txt", chunksize=1000, index=False, sep="\t")


def preprocess_es_eval_data():
    from collections import defaultdict
    import numpy as np

    print("Pre-processing Spanish lexical decision data")
    spanish_data = pd.read_csv("eval/raw/es.txt", delimiter=",", header=0)
    print(len(spanish_data))
    spanish_data = spanish_data.dropna().reset_index(drop=True)
    print("After dropping NaNs: ")

    spanish_data["spelling"] = spanish_data["spelling"].astype("string")
    spanish_data["rt"] = spanish_data["rt"].astype("int")
    spanish_data["accuracy"] = spanish_data["accuracy"].astype("int")
    # Adjust different coding of non-words across languages
    spanish_data["lexicality"] = spanish_data["lexicality"].replace("NW", "N")
    # There is a non-word stimulus "nan"
    spanish_data['spelling'] = spanish_data['spelling'].fillna("nan")

    print(len(spanish_data))

    spanish_data = spanish_data
    lexicality = defaultdict(list)
    rts = defaultdict(list)
    accuracies = defaultdict(list)

    lower_threshold = spanish_data["rt"].quantile(0.01)
    upper_threshold = spanish_data["rt"].quantile(0.99)
    i = 0
    for i, row in enumerate(spanish_data.itertuples()):
        token = row.spelling
        # The spanish data contains a lot of outliers, so I am cutting the values below the first and the last percentile
        # For the record: Percentil, Value: [0.001, 20.0], [0.005,  157.0] , [0.01,  484.0] , [0.05,  620.0] , [0.1,  688.0] , [0.9,  2839.0] , [0.99,  7753.0] , [0.995,  11125.0] , [0.999,  26345.68100000173]
        if row.rt > lower_threshold and row.rt < upper_threshold:
            rts[token].append(row.rt)
            accuracies[token].append(row.accuracy)
            lexicality[token].append(row.lexicality)

        # if i % 500000 == 0:
        #     print(i, len(lexicality))
    assert (len(lexicality.keys()) == len(accuracies.keys()) == len(rts.keys()))

    with open("eval/es.txt", "w", encoding="utf-8") as outfile:
        outfile.write("spelling\tlexicality\trt\taccuracy\n")
        for token in lexicality.keys():

            try:
                av_rt = round(np.mean(np.asarray(rts[token])), 2)
                av_acc = np.nanmean(np.asarray(accuracies[token]))
                lex = set(lexicality[token])
                if len(lex) > 1:
                    print(token, lex)
                outfile.write("\t".join([token, lex.pop(), str(av_rt), str(av_acc)]))
                outfile.write("\n")

            except TypeError as e:
                print(e)
                print(token, lex, rts[token], accuracies[token])


def get_eval_data():

    # Get Dutch Data
    # Note: this is not the most recent version of the dataset but it is the one we used.
    # Check here http://crr.ugent.be/programs-data/lexicon-projects for a more recent one with twice as many stimuli

    # print("Get Dutch lexical decision data")
    # url = "http://crr.ugent.be/dlp/txt/dlp-items.txt.zip"
    # r = requests.get(url)
    # z = zipfile.ZipFile(io.BytesIO(r.content))
    # src = "dlp-items.txt"
    # dest = "eval/nl.txt"
    # z.getinfo(src).filename = dest
    # z.extract(src)

    # # Get English Data
    # print("Get English lexical decision data")
    # url = "http://crr.ugent.be/blp/txt/blp-items.txt.zip"
    # r = requests.get(url)
    # z = zipfile.ZipFile(io.BytesIO(r.content))
    # src = "blp-items.txt"
    # dest = "eval/en-new.txt"
    # z.getinfo(src).filename = dest
    # z.extract(src)

    # Get French Data
    print("Get French lexical decision data")
    url = "https://static-content.springer.com/esm/art%3A10.3758%2FBRM.42.2.488/MediaObjects/Ferrand-BRM-2010.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    print("Save")
    z.extractall("eval/raw/")
    preprocess_fr_eval_data()

    # Get Spanish Data
    print("Get Spanish lexical decision data")
    url = "https://figshare.com/ndownloader/files/11209613"
    r = requests.get(url)
    open("eval/raw/es.txt", "w", encoding="utf-8").write(r.text)
    preprocess_es_eval_data()


def download_and_extract(url, outputdir, lang):
    # check if already done
    txtfname = outputdir + "/" + lang + ".txt"
    if os.path.exists(txtfname):
        return True

    print(url)
    response = requests.get(url, stream=True)

    if not response.status_code == 404:
        langdir = outputdir + lang
        os.makedirs(langdir, exist_ok=True)
        file = tarfile.open(fileobj=response.raw, mode="r|gz")
        file.extractall(path=langdir)
        print("Cleaning up file structure")
        for d in os.listdir(langdir):
            path = langdir + "/" + d
            print(path)
            for file in os.listdir(path):
                filepath = path + "/" + file

                if not file.endswith("sentences.txt"):
                    #print("deleting: "+ file)
                    os.remove(filepath)

                else:
                    #print("renaming: " + filepath)
                    os.rename(filepath, txtfname)

        shutil.rmtree(langdir, ignore_errors=True)

        return True
    else:
        print(response)
        return False


def get_training_data():
    prefix = "https://downloads.wortschatz-leipzig.de/corpora/"
    used_corpora = {"eng": "eng_news_2020_100K-sentences.txt", "mon": "mon_news_2020_100K-sentences.txt", "fra": "fra_news_2022_100K-sentences.txt", "nld": "nld_news_2022_100K-sentences.txt", "spa": "spa_news_2022_100K-sentences.txt", "deu": "deu_news_2022_100K-sentences.txt", "cat": "cat_news_2022_100K-sentences.txt", "ces": "ces_news_2022_100K-sentences.txt", "fin": "fin_news_2022_100K-sentences.txt", "hun": "hun_news_2022_100K-sentences.txt", "ita": "ita_news_2022_100K-sentences.txt", "pol": "pol_news_2022_100K-sentences.txt", "por": "por_news_2022_100K-sentences.txt", "rus": "rus_news_2022_100K-sentences.txt", "swe": "swe_news_2022_100K-sentences.txt", "hbs": "hbs_mixed_2014_100K-sentences.txt"}
    for lang, corpus in used_corpora.items():
        print("Getting training data for " + lang)
        corpus = corpus.replace("-sentences.txt", ".tar.gz")
        download_and_extract(prefix+corpus, "train/", lang )

def get_derivational_data():
    prefix = "https://github.com/kbatsuren/MorphyNet/"
    res = requests.get(prefix)
    soup = bs(res.text, features="html.parser")
    langs = soup.find_all('a', class_="js-navigation-open")

    for langdir in langs:
        lang = langdir.text
        outdir = "morphynet2/" + lang + "/"
        os.makedirs(outdir, exist_ok=True)
        if not lang == "README.md":

            filename = lang + ".derivational.v1.tsv"
            rawpath = prefix + "blob/main/" + lang + "/" + filename  + "?raw=true"
            print("Downloading from: " + rawpath)
            response = requests.get(rawpath)
            data = response.text

            with open(outdir + filename, 'w', encoding="utf=8") as file:
                file.write(data)



get_training_data()
get_eval_data()
get_derivational_data()

