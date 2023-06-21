import pandas as pd

import seaborn
import matplotlib.pyplot as plt
# TODO: Improve the plot for figure 4 in the paper
# Settings
sizes = [100000]
vocab_sizes =[500, 1000, 2500, 5000, 7500, 10000, 20000,30000,40000,50000 ]
languages = ["hun", "por", "cat", "fin", "ita", "rus", "ces", "fra", "eng", "mon", "spa", "deu", "hbs", "pol", "swe"]
algs = [ "WPC", "UNI"]
traindir = "../../data/train/"
seed = 42

# Collect results
language_results =[]
for language in languages:
    results = pd.read_csv("../../results/derivational_overlap/overlap_monolingual/" + language  + ".csv")
    language_results.append(results)

results_frame = pd.concat([x for x in language_results])

# Plotting
palette = "vivid"
hue_order = ["hun", "fin", "cat", "swe","deu", "fra", "spa","por", "hbs", "mon", "eng",  "ita", "rus", "ces",  "pol"]
markers=["*", "*", "v", "o", "o", "v", "v", "v", "p", "*", "o", "v", "p", "p", "p"]
markers1=["*", "*", "*", "*", "v", "v", "v", "v"]
markers2=["*", "*", "*", "v", "v", "v"]
marker_lookup ={"UNI":"*", "WPC":"^"}
families ={"hun":"Other", "fin":"Other", "cat":"Romanic", "swe":"Germanic","deu":"Germanic", "fra":"Romanic", "spa":"Romanic","por":"Romanic", "hbs":"Slavic", "mon":"Other", "eng":"Germanic","ita":"Romanic", "rus":"Slavic", "ces":"Slavic",  "pol":"Slavic"}
# Making some selections

limited_data = results_frame[results_frame["training_size"] == 100000]
limited_data =limited_data[limited_data["algorithm"].isin(algs) ]
limited_data['language_family'] = limited_data['language'].map(families)
# limited_data_uni = limited_data[limited_data["algorithm"]=="UNI" ]
# limited_data_wpc= limited_data[limited_data["algorithm"]=="WPC" ]
limited_data_gersla = limited_data[limited_data["language_family"].isin(["Germanic", "Slavic"])]
limited_data_rest = limited_data[limited_data["language_family"].isin(["Romanic", "Other"])]
print(limited_data)
# TODO : This plot is not very well visualized. Need to adapt.
# ax = seaborn.lmplot(limited_data, x= "vocab_size", y="overlap",palette=palette, hue="language", col = "algorithm",  fit_reg=False, markers=markers, hue_order =hue_order)
fig, axes = plt.subplots(2,2, figsize=(10,7), sharex=True, sharey=True)
axes = axes.flatten()
limited_data = limited_data.rename(columns={'language': 'Language', 'algorithm': 'Algorithm'})
germanic = limited_data[limited_data["Language"].isin(["swe", "deu", "eng"])]
romanic = limited_data[limited_data["Language"].isin(["ita", "fra", "spa", "por", "cat"])]
slavic = limited_data[limited_data["Language"].isin(["rus", "ces", "pol", "hbs"])]
other= limited_data[limited_data["Language"].isin(["hun", "fin", "mon"])]

mypalette =seaborn.color_palette("tab20")
mypalette = mypalette[0:11]+mypalette[13:14]+ mypalette[16:]
seaborn.scatterplot(data=germanic, x= "vocab_size",  y="overlap", hue="Language", hue_order=["swe","deu", "eng"],style="Algorithm", markers = marker_lookup, palette=mypalette[0:3],  ax=axes[0], s=200, edgecolor="dimgrey")

seaborn.scatterplot(data=romanic, x= "vocab_size",  y="overlap", hue="Language", style="Algorithm", markers = marker_lookup, palette=mypalette[3:8], hue_order =["cat","spa", "fra", "ita","por"], ax=axes[1], s=200, edgecolor="dimgrey")

seaborn.scatterplot(data=slavic, x= "vocab_size",  y="overlap", hue="Language", hue_order=["hbs","ces","rus", "pol"], style="Algorithm", markers = marker_lookup, palette=mypalette[8:12],  ax=axes[2],s=200, edgecolor="dimgrey")

seaborn.scatterplot(data=other, x= "vocab_size",  y="overlap", hue="Language", style="Algorithm", markers = marker_lookup, palette=mypalette[12:15],  ax=axes[3],s=200, edgecolor="dimgrey")

# Legend
handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(handles=handles[1:6], labels=labels[1:6])
print(labels)
handles, labels = axes[2].get_legend_handles_labels()
axes[2].legend(handles=handles[1:5], labels=labels[1:5])
print(labels)
handles, labels = axes[3].get_legend_handles_labels()
axes[3].legend(handles=handles[1:4], labels=labels[1:4])
print(labels)

# Axes
axes[1].yaxis.set_visible(False)
axes[3].yaxis.set_visible(False)
axes[0].spines[['right', 'top']].set_visible(False)
axes[1].spines[["left",'right', 'top']].set_visible(False)
axes[2].spines[['right', 'top']].set_visible(False)
axes[3].spines[["left",'right', 'top']].set_visible(False)

# Labels
axes[2].set_xlabel("Vocabulary Size")
axes[3].set_xlabel("Vocabulary Size")
axes[0].set_ylabel("Derivational Coverage")
axes[2].set_ylabel("Derivational Coverage")
axes[0].title.set_text('Germanic')
axes[1].title.set_text('Romanic')
axes[2].title.set_text('Slavic')
axes[3].title.set_text('Other')
# axes[2].set_xlabel("")
# axes[3].set_xlabel("")
#
#
# axes[0].set_ylabel(r"Corr. Accuracy")
# axes[2].set_ylabel(r"Corr. Reading Time")
#
#seaborn.scatterplot(data=limited_data_gersla, x= "vocab_size",  y="overlap",palette="muted", hue="language", col ="language_family", col_order=["Germanic",  "Slavic"])
#seaborn.lmplot(data=limited_data_rest, x= "vocab_size",  y="overlap",palette="muted", hue="language", col ="language_family", col_order=["Germanic",  "Slavic"])
# seaborn.catplot(data=limited_data_wpc, x= "vocab_size",  y="overlap",palette=palette, hue="language", col ="language_family", ax=ax)
plt.show()
fig.savefig("../../results/derivational_overlap/derivational.png")

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



