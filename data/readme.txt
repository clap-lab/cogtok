Training data are the sentences-files from the most recent 100k news subset from the Leipzig collection:
https://wortschatz.uni-leipzig.de/en/download/

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


The respective word files are used for training morfessor.


Eval files are lexical decision data from:
Dutch: http://crr.ugent.be/papers/dlp2_items.tsv
English: http://crr.ugent.be/blp/txt/blp-items.txt.zip
French: https://static-content.springer.com/esm/art%3A10.3758%2FBRM.42.2.488/MediaObjects/Ferrand-BRM-2010.zip
Spanish: https://figshare.com/articles/dataset/Lexical_decision_data/5924647

French and Spanish require some pre-processing, see eval/preprocessing/preprocess_es_eval.py and preprocess_fr_eval.py
