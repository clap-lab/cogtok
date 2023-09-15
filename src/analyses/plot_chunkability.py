
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

### Figure 1
# Data
# results = pd.read_csv("subresults/chunkability.csv")
# words = results[results["Category"]=="Words"]
# nonwords = results[results["Category"]=="Non-Words"]
#
# fig, axes = plt.subplots(2,2, sharex=True, sharey =True)
# axes = axes.flatten()
#
# # Colors
# palette = ["mediumpurple", "mediumturquoise", "midnightblue", "lightgrey"]
# hue_order =["UNI", "BPE", "WPC", "Length"]
#
# # Bar Plots
# seaborn.barplot(data=words[words["Metric"] == "Accuracy"],  x= "Language", y="Correlation",hue="Model", hue_order = hue_order, palette=palette , ax=axes[0],)
# seaborn.barplot(data=nonwords[nonwords["Metric"] == "Accuracy"], x= "Language", y="Correlation",hue="Model",palette=palette , hue_order = hue_order, ax=axes[1])
# seaborn.barplot(data= words[words["Metric"] == "Reading Time"],  x= "Language", y="Correlation",hue="Model",palette=palette , hue_order = hue_order,  ax=axes[2])
# seaborn.barplot(data= nonwords[nonwords["Metric"] == "Reading Time"],x= "Language", y="Correlation",hue="Model", hue_order = hue_order,palette=palette ,  ax=axes[3])
#
# # Labels
# # fig.suptitle("Chunkability")
# axes[0].set_xlabel("Words")
# axes[1].set_xlabel("Non-Words")
# axes[2].set_xlabel("")
# axes[3].set_xlabel("")
#
#
# axes[0].set_ylabel(r"Corr. Accuracy")
# axes[2].set_ylabel(r"Corr. Reading Time")
# # Axes
# axes[0].tick_params(labelbottom=True)
# axes[1].tick_params(labelbottom=True)
# axes[0].xaxis.set_label_position('top')
# axes[1].xaxis.set_label_position('top')
# plt.setp(axes[2].get_xticklabels(), visible=False)
# plt.setp(axes[3].get_xticklabels(), visible=False)
#
#
# axes[1].yaxis.set_visible(False)
# axes[3].yaxis.set_visible(False)
# axes[0].spines[['right', 'top']].set_visible(False)
# axes[1].spines[["left",'right', 'top']].set_visible(False)
# axes[2].spines[['right', 'top']].set_visible(False)
# axes[3].spines[["left",'right', 'top']].set_visible(False)
#
# # Legend
# axes[0].get_legend().remove()
# axes[1].get_legend().remove()
# axes[2].get_legend().remove()
# handles, labels = axes[3].get_legend_handles_labels()
# axes[3].legend(handles=handles, labels=labels)
# seaborn.move_legend(axes[3], "lower right")
# plt.legend(fontsize='x-small')
#
#
#
# fig.tight_layout()
# plt.show()
# fig.savefig("../../results/plots/chunkability.png")

### Figure 1
# Data
# results = pd.read_csv("subresults/chunkability.csv")
# words = results[results["Category"]=="Words"]
# nonwords = results[results["Category"]=="Non-Words"]
#
# fig, axes = plt.subplots(2,2, sharex=True, sharey =True)
# axes = axes.flatten()
#
# # Colors
# palette = ["mediumpurple", "mediumturquoise", "midnightblue", "lightgrey"]
# hue_order =["UNI", "BPE", "WPC", "Length"]
#
# # Bar Plots
# seaborn.barplot(data=words[words["Metric"] == "Accuracy"],  x= "Language", y="Correlation",hue="Model", hue_order = hue_order, palette=palette , ax=axes[0],)
# seaborn.barplot(data=nonwords[nonwords["Metric"] == "Accuracy"], x= "Language", y="Correlation",hue="Model",palette=palette , hue_order = hue_order, ax=axes[1])
# seaborn.barplot(data= words[words["Metric"] == "Reading Time"],  x= "Language", y="Correlation",hue="Model",palette=palette , hue_order = hue_order,  ax=axes[2])
# seaborn.barplot(data= nonwords[nonwords["Metric"] == "Reading Time"],x= "Language", y="Correlation",hue="Model", hue_order = hue_order,palette=palette ,  ax=axes[3])
#
# # Labels
# # fig.suptitle("Chunkability")
# axes[0].set_xlabel("Words")
# axes[1].set_xlabel("Non-Words")
# axes[2].set_xlabel("")
# axes[3].set_xlabel("")
#
#
# axes[0].set_ylabel(r"Corr. Accuracy")
# axes[2].set_ylabel(r"Corr. Reading Time")
# # Axes
# axes[0].tick_params(labelbottom=True)
# axes[1].tick_params(labelbottom=True)
# axes[0].xaxis.set_label_position('top')
# axes[1].xaxis.set_label_position('top')
# plt.setp(axes[2].get_xticklabels(), visible=False)
# plt.setp(axes[3].get_xticklabels(), visible=False)
#
#
# axes[1].yaxis.set_visible(False)
# axes[3].yaxis.set_visible(False)
# axes[0].spines[['right', 'top']].set_visible(False)
# axes[1].spines[["left",'right', 'top']].set_visible(False)
# axes[2].spines[['right', 'top']].set_visible(False)
# axes[3].spines[["left",'right', 'top']].set_visible(False)
#
# # Legend
# axes[0].get_legend().remove()
# axes[1].get_legend().remove()
# axes[2].get_legend().remove()
# handles, labels = axes[3].get_legend_handles_labels()
# axes[3].legend(handles=handles, labels=labels)
# seaborn.move_legend(axes[3], "lower right")
# plt.legend(fontsize='x-small')
#
#
#
# fig.tight_layout()
# plt.show()
# fig.savefig("../../results/plots/chunkability.png")
### Figure 2
# Data
# results = pd.read_csv("subresults/vocab_size.csv", index_col=0)
# seaborn.set(font_scale=1.4, style="white")
# palette2 = ["mediumpurple", "goldenrod", "midnightblue", "seagreen"]
# dashes = [(4,1 ),  (3, 5), (1, 1),(2, 1)]
# markers = ["o", "D", "s", "v"]
# # languages = ["English", "Dutch", "French", "Spanish"]
# plt.figure(figsize=(10,4))
# ax = seaborn.lineplot( data=results, palette=palette2, dashes=dashes, markers=markers, linewidth=3, hue_order=["English","Spanish", "Dutch",  "French"] )
#
#
# ax.set(xlabel='Vocabulary Size', ylabel="Pearson's Correlation")
#
#
# plt.show()
# fig=ax.get_figure()
# fig.tight_layout()
# fig.savefig("../../results/plots/vocab_curve.png")

### Figure 3: grouped bar plot
# seaborn.set(font_scale=1.4, style="white")
# plt.figure(figsize=(10,4))
# palette3 = ["mediumpurple", "mediumturquoise", "midnightblue", "seagreen"]
# results = pd.read_csv("subresults/pretrained_multilingual_vocab.csv")
# ax = seaborn.catplot( data=results, x= "Model", y="Correlation",kind ="bar", hue="Language", col="Metric", palette=palette3 , legend=False)
#
# plt.legend(loc ="lower right")
# ax.set_titles('{col_name}')
# ax.set(xlabel='', ylabel="Pearson's Correlation")
# ax.tight_layout()
# plt.show()
# ax.savefig("../../results/plots/pretrained_multilingual.png")

### Figure Appendix: fourlingual models
#Data
# results = pd.read_csv("subresults/fourlingual.csv")
# words = results[results["Category"]=="Words"]
# nonwords = results[results["Category"]=="Non-Words"]
#
# fig, axes = plt.subplots(2,2, sharex=True, sharey =True)
# axes = axes.flatten()
#
# # Colors
# palette = ["goldenrod", "mediumturquoise", "midnightblue", "lightgrey"]
# hue_order =["mono_50k", "multi_50k","multi_70k"]
#
# # Bar Plots
# seaborn.barplot(data=words[words["Metric"] == "Accuracy"],  x= "Language", y="Correlation",hue="Model", hue_order = hue_order, palette=palette , ax=axes[0],)
# seaborn.barplot(data=nonwords[nonwords["Metric"] == "Accuracy"], x= "Language", y="Correlation",hue="Model",palette=palette , hue_order = hue_order, ax=axes[1])
# seaborn.barplot(data= words[words["Metric"] == "Reading Time"],  x= "Language", y="Correlation",hue="Model",palette=palette , hue_order = hue_order,  ax=axes[2])
# seaborn.barplot(data= nonwords[nonwords["Metric"] == "Reading Time"],x= "Language", y="Correlation",hue="Model", hue_order = hue_order,palette=palette ,  ax=axes[3])
#
# # Labels
# # fig.suptitle("Chunkability")
# axes[0].set_xlabel("Words")
# axes[1].set_xlabel("Non-Words")
# axes[2].set_xlabel("")
# axes[3].set_xlabel("")
#
#
# axes[0].set_ylabel(r"Corr. Accuracy")
# axes[2].set_ylabel(r"Corr. Reading Time")
# # Axes
# axes[0].tick_params(labelbottom=True)
# axes[1].tick_params(labelbottom=True)
# axes[0].xaxis.set_label_position('top')
# axes[1].xaxis.set_label_position('top')
# plt.setp(axes[2].get_xticklabels(), visible=False)
# plt.setp(axes[3].get_xticklabels(), visible=False)
#
#
# axes[1].yaxis.set_visible(False)
# axes[3].yaxis.set_visible(False)
# axes[0].spines[['right', 'top']].set_visible(False)
# axes[1].spines[["left",'right', 'top']].set_visible(False)
# axes[2].spines[['right', 'top']].set_visible(False)
# axes[3].spines[["left",'right', 'top']].set_visible(False)
#
# # Legend
# axes[0].get_legend().remove()
# axes[1].get_legend().remove()
# axes[2].get_legend().remove()
# handles, labels = axes[3].get_legend_handles_labels()
# axes[3].legend(handles=handles, labels=labels)
# seaborn.move_legend(axes[3], "lower right")
# plt.legend(fontsize='x-small')
#
# fig.tight_layout()
# plt.show()
# fig.savefig("../../results/plots/fourlingual.png")

### Figure Appendix: Figure 1 with Num_Splits instead
# Data
results = pd.read_csv("subresults/num_splits.csv")
words = results[results["Category"]=="Words"]
nonwords = results[results["Category"]=="Non-Words"]

fig, axes = plt.subplots(2,2, sharex=True, sharey =True)
axes = axes.flatten()

# Colors
palette = ["mediumpurple", "mediumturquoise", "midnightblue", "lightgrey"]
hue_order =["UNI", "BPE", "WPC", "Length"]

# Bar Plots
seaborn.barplot(data=words[words["Metric"] == "Accuracy"],  x= "Language", y="Correlation",hue="Model", hue_order = hue_order, palette=palette , ax=axes[0],)
seaborn.barplot(data=nonwords[nonwords["Metric"] == "Accuracy"], x= "Language", y="Correlation",hue="Model",palette=palette , hue_order = hue_order, ax=axes[1])
seaborn.barplot(data= words[words["Metric"] == "Reading Time"],  x= "Language", y="Correlation",hue="Model",palette=palette , hue_order = hue_order,  ax=axes[2])
seaborn.barplot(data= nonwords[nonwords["Metric"] == "Reading Time"],x= "Language", y="Correlation",hue="Model", hue_order = hue_order,palette=palette ,  ax=axes[3])

# Labels
fig.suptitle("Num_Splits")
axes[0].set_xlabel("Words")
axes[1].set_xlabel("Non-Words")
axes[2].set_xlabel("")
axes[3].set_xlabel("")


axes[0].set_ylabel(r"Corr. Accuracy")
axes[2].set_ylabel(r"Corr. Reading Time")
# Axes
axes[0].tick_params(labelbottom=True)
axes[1].tick_params(labelbottom=True)
axes[0].xaxis.set_label_position('top')
axes[1].xaxis.set_label_position('top')
plt.setp(axes[2].get_xticklabels(), visible=False)
plt.setp(axes[3].get_xticklabels(), visible=False)


axes[1].yaxis.set_visible(False)
axes[3].yaxis.set_visible(False)
axes[0].spines[['right', 'top']].set_visible(False)
axes[1].spines[["left",'right', 'top']].set_visible(False)
axes[2].spines[['right', 'top']].set_visible(False)
axes[3].spines[["left",'right', 'top']].set_visible(False)

# Legend
axes[0].get_legend().remove()
axes[1].get_legend().remove()
axes[2].get_legend().remove()
handles, labels = axes[3].get_legend_handles_labels()
axes[3].legend(handles=handles, labels=labels)
seaborn.move_legend(axes[3], "lower right")
plt.legend(fontsize='x-small')



fig.tight_layout()
plt.show()
fig.savefig("../../results/plots/num_splits.png")