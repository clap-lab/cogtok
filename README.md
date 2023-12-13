This repository contains the code for the following paper.<br>
- Lisa Beinborn, Yuval Pinter (EMNLP 2023): <br>
["Analyzing Cognitive Plausibility of Subword Tokenization"](https://aclanthology.org/2023.emnlp-main.272)

Please check the paper for all the references and cite them appropriately when using this code. 

If you want to rerun everything, follow these steps: 
- install the packages in requirements.txt
- run *data/get_data.py*
- run *src/train_models.py* (and optionally also *calculate_derivational_overlap.py*, if you are interested)
- run *src/analyses/evaluate_with_lexdec_data.py* (and optionally also the other analyses in the directory )

You can also work directly with our already trained models. They can be found in *results/trained/models*. 

The plots from the paper can be found in *results/plots*.

If you have questions, contact l.b\*\*nb\*\*n@vu.nl

## Citation Format

Please cite this work as:
```
@inproceedings{beinborn-pinter-2023-analyzing,
    title = "Analyzing Cognitive Plausibility of Subword Tokenization",
    author = "Beinborn, Lisa  and
      Pinter, Yuval",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.272",
    pages = "4478--4486",
}
```

