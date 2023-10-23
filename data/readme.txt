If you run get_data.py, you download the following datasets:

For training the tokenizers, we used sentences-files from the most recent 100k news subset from the Leipzig collection: https://wortschatz.uni-leipzig.de/en/download/

The exact file names are listed below. 
2020
ENG: eng_news_2020_100K-sentences.txt
MON: mon_news_2020_100K-sentences.txt

2022
FRA: fra_news_2022_100K-sentences.txt
NLD: nld_news_2022_100K-sentences.txt
SPA: spa_news_2022_100K-sentences.txt
DEU: deu_news_2022_100K-sentences.txt
CAT: cat_news_2022_100K-sentences.txt
CES: ces_news_2022_100K-sentences.txt
FIN: fin_news_2022_100K-sentences.txt
HUN: hun_news_2022_100K-sentences.txt
ITA: ita_news_2022_100K-sentences.txt
POL: pol_news_2022_100K-sentences.txt
POR: por_news_2022_100K-sentences.txt
RUS: rus_news_2022_100K-sentences.txt
SWE: swe_news_2022_100K-sentences.txt

OTHER:
HBS: hbs_mixed_2014_100K-sentences.txt


For evaluation, we used lexical decision data from the following sources:
Dutch: http://crr.ugent.be/papers/dlp2_items.tsv
English: http://crr.ugent.be/blp/txt/blp-items.txt.zip
French: https://static-content.springer.com/esm/art%3A10.3758%2FBRM.42.2.488/MediaObjects/Ferrand-BRM-2010.zip
Spanish: https://figshare.com/articles/dataset/Lexical_decision_data/5924647

For the French and Spanish files, we applied some pre-processing to aggregate the signals.

For the morphological analysis, we used the derivational data from morphynet:
https://github.com/kbatsuren/MorphyNet

