import os
import pandas
path = "wordlists"
for file in os.listdir(path):
    print(file)
    fullpath = path + "/" + file
    if os.path.isfile(fullpath) and not file.startswith("."):
        with open(fullpath + "_processed", "w") as outfile:
            for line in open(path + "/" + file, "r"):
                index, word, frequency = line.split("\t")
                outfile.write(word + "\t" + frequency)