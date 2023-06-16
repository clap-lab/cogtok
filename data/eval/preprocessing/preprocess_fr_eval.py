import pandas as pd
import numpy as np
nonwords = pd.read_excel("fr/FLP-pseudowords.xls", index_col=None, header =0)
nonwords.reset_index(drop=True, inplace=True)
words = pd.read_excel("fr/FLP-words.xls", index_col=None, header =0, usecols = range(7))
words.reset_index(drop=True, inplace=True)
words["lexicality"] = ["W" for x in range(len(words))]
nonwords["lexicality"] = ["N" for x in range(len(nonwords))]
# concat the two frames, add lexicality value based on source dataset
french_data = pd.concat([nonwords, words],axis =0)


#Columns: item, n_trials, err, rt, sd, rtz, n_used
# Accuracy is 1-err
french_data["accuracy"] = (1-french_data["err"]).round(decimals=2)
french_data["rt"] = french_data["rt"].round(decimals=2)
print(french_data.head().to_string())
french_data = french_data.rename(columns={"item":"spelling"})
mapped_data = french_data[["spelling","lexicality","rt","accuracy"]]
# with open("../fr.txt", "w") as outfile:
#     for i, row in enumerate(mapped_data):
#         if i % 100 == 0:
#             print(i, len(mapped_data))
#             print("\t".join(list(row)))
#         outfile.write("\t".join(list(row)))
print("Writing it out")
mapped_data.to_csv("../fr.txt", chunksize=1000, index =False, sep="\t")
