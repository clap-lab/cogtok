import pandas as pd
from collections import defaultdict
import numpy as np

print("reading data")
spanish_data = pd.read_csv("es_unprocessed.txt", delimiter=",", header=0)
print(len(spanish_data))
spanish_data = spanish_data.dropna().reset_index(drop=True)
print("After dropping NaNs: ")
print(len(spanish_data))
spanish_data["spelling"]= spanish_data["spelling"].astype("string")
spanish_data["rt"]= spanish_data["rt"].astype("int")
spanish_data["accuracy"]= spanish_data["accuracy"].astype("int")
# Adjust different coding of non-words across languages
spanish_data["lexicality"] = spanish_data["lexicality"].replace("NW", "N")
# There is a non-word stimulus "nan"
spanish_data['spelling'] = spanish_data['spelling'].fillna("nan")

print("done reading data")
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
    # For the recort: Percentil, Value: [0.001, 20.0], [0.005,  157.0] , [0.01,  484.0] , [0.05,  620.0] , [0.1,  688.0] , [0.9,  2839.0] , [0.99,  7753.0] , [0.995,  11125.0] , [0.999,  26345.68100000173]
    if row.rt > lower_threshold and row.rt < upper_threshold:
        rts[token].append(row.rt)
        accuracies[token].append(row.accuracy)
        lexicality[token].append(row.lexicality)

    if i%500000 == 0:
        print(i, len(lexicality))
assert(len(lexicality.keys()) == len(accuracies.keys()) == len(rts.keys()))



# This was for exploration
#all_rts = pd.Series([int(rt) for tokenrts in rts.values() for rt in tokenrts])
# print("Statistics for reading times: ")
# print(all_rts.describe())
# print(all_rts.median())
# print("0.001",all_rts.quantile(0.001))
# print("0.005",all_rts.quantile(0.005))
#
# print("0.01",all_rts.quantile(0.01))
# print("0.05",all_rts.quantile(0.05))
# print("0.1",all_rts.quantile(0.1))
# print("0.9",all_rts.quantile(0.9))
# print("0.99",all_rts.quantile(0.99))
# print("0.995",all_rts.quantile(0.995))
# print("0.999",all_rts.quantile(0.999))
# print(all_rts.quantile(0.9))
# print(all_rts[0:500])
# sorted = all_rts.sort_values(ascending=True)
# sorted = sorted.reset_index(drop=True)
# print(sorted[0:500])
# print(sorted[-500:])
# ax = sorted.plot(style='.')
# ax.figure.savefig('spanish_rts.png')
#
# second = sorted.plot.hist(bins=10)
# second.figure.savefig('spanish_rts_hist.png')

with open("../es.txt", "w") as outfile:
    outfile.write("spelling\tlexicality\trt\taccuarcy\n")
    for token in lexicality.keys():

        try:
            av_rt = round(np.mean(np.asarray(rts[token])), 2)
            av_acc = np.nanmean(np.asarray(accuracies[token]))
            lex = set(lexicality[token])
            if len(lex)>1:
                print(token, lex)
            outfile.write("\t".join([token, lex.pop(), str(av_rt), str(av_acc)]))
            outfile.write("\n")

        except TypeError as e:
            print(e)
            print(token, lex, rts[token], accuracies[token])



