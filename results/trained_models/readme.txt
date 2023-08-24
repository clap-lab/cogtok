Important note: 
The output.csv files in this folder contain the columns "reading time" and "accuracy". After I generated them, I adjusted the Spanish evaluation data to remove the extreme outlier responses (see data/eval/preprocessing). That is why these columns do not contain the correct values anymore. For evaluation, I use the preprocessed evaluation files.

TODO: it would be cleaner to remove these columns from the model output but I can't do it with bash because the comma is used as the field separator but also in the lists of splits in the second column. With pandas, it will take forever which is why I haven't done it yet. 


Fourlingual means that the model has been trained on the English, French, Spanish, and Dutch data jointly. 
