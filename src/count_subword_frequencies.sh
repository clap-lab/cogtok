#!/bin/bash
fullfile='/Users/lisabeinborn/Research/tokenization/results/models/BPE_wiki_10000_vocabulary.txt'
outfile='/Users/lisabeinborn/Research/tokenization/results/models/BPE_wiki_10000_frequencies.txt'
data='/Users/lisabeinborn/Data/wikitext-103-raw/wiki.train.raw'
touch ${outfile}
while read p; do
  echo -n $p
  grep "$p" ${data} |wc -l
done < ${fullfile} > ${outfile}
