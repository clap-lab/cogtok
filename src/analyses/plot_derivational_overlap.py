import pandas as pd

import seaborn

# TODO: Improve the plot for figure 4 in the paper
# Settings
sizes = [100000]
vocab_sizes =[500, 1000, 2500, 5000, 7500, 10000, 20000,30000,40000,50000 ]
languages = ["hun", "por", "cat", "fin", "ita", "rus", "ces", "fra", "eng", "mon", "spa", "deu", "hbs", "pol", "swe"]
algs = ["BPE", "WPC", "UNI"]
traindir = "../../data/train/"
seed = 42

# Collect results
language_results =[]
for language in languages:
    results = pd.read_csv("../../results/derivational_overlap/" + language  + ".csv")
    language_results.append(results)

results_frame = pd.concat([x for x in language_results])

# Plotting
palette = "rainbow"
hue_order = ["hun", "fin", "cat", "swe","deu", "fra", "spa","por", "hbs", "mon", "eng",  "ita", "rus", "ces",  "pol"]
markers=["*", "*", "v", "o", "o", "v", "v", "v", "p", "*", "o", "v", "p", "p", "p"]

# Making some selections
limited_data = results_frame[results_frame["training_size"] == 100000]
limited_data = results_frame[results_frame["algorithm"] == algs[0]]

# TODO : This plot is not very well visualized. Need to adapt.
ax = seaborn.lmplot(limited_data, x= "vocab_size", y="overlap",palette=palette, hue="language", col = "algorithm",  fit_reg=False, markers=markers, hue_order = hue_order)
ax.figure.savefig("../results/derivational_overlap/only_wpc.png")

# Note: I also calculated coverage for cross-lingual trained_models, see results/derivational_overlap/overlap_crosslingualmodels.xlsx







# Tried out plot variants here
# # norm = plt.Normalize(results_frame["overlap"].min(), results_frame["overlap"].max())
# # sm = plt.cm.ScalarMappable(cmap=palette, norm=norm)
# # sm.set_array([])
#
#ax = seaborn.scatterplot(results_frame, x= "vocab_size", y="overlap",palette=palette, hue="language",  style="training_size", markers=[".", "*", "p", "o"], alpha=0.8)
#ax = seaborn.lmplot(results_frame, x= "vocab_size", y="overlap",palette=palette, hue="language",  col="training_size", fit_reg=False)
# #ax = seaborn.scatterplot(results_frame[results_frame["training_size"]==50000], x= "vocab_size", y="overlap",palette=palette, hue="language", markers="+")
# # ax = seaborn.lmplot(results_frame[results_frame["training_size"]==75000], x= "vocab_size", y="overlap",palette=palette, hue="language", fit_reg=False, markers="p")
#ax = seaborn.stripplot(results_frame, x= "vocab_size", y="overlap",palette=palette, hue="language", dodge=True)
# # ax.get_legend().remove()
# # ax.figure.colorbar(sm)



