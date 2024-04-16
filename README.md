This repository contains the code for the following paper.<br>
- Lisa Beinborn, Yuval Pinter (EMNLP 2023): <br>
["Analyzing Cognitive Plausibility of Subword Tokenization"](https://aclanthology.org/2023.emnlp-main.272)

Please check the paper for all the references and cite them appropriately when using this code. 

If you want to rerun everything, follow these steps:
- clone repository and cd into it via
```bash
git clone git@github.com:clap-lab/cogtok
cd cogtok
```
- install the packages in requirements.txt via
```bash
pip install -r requirements.txt`
```
- Download the data: 
```bash
python3 data/get_data.py
``````
- Train the models: 
```bash
python3 src/train_models.py
```
- (and optionally also `python src/calculate_derivational_overlap.py`, if you are interested)
- Evaluate with the lexical decision data:
```bash
`python3 src/analyses/evaluate_with_lexdec_data.py`
```
- Have a look at the other analyses in the directory, if you are interested. 

You can also work directly with our already trained models. They can be found in *results/trained/models*. 

The plots from the paper can be found in *results/plots*.

If you have questions, contact lisa.b\*\*nb\*\*n@uni-goettingen.de

## Citation Format

Please cite this work as:
```
@inproceedings{beinborn-pinter-2023-analyzing,
    title = "Analyzing Cognitive Plausibility of Subword Tokenization",
    author = "Beinborn, Lisa and Pinter, Yuval",
    editor = "Bouamor, Houda and Pino, Juan and Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.272",
    pages = "4478--4486",
}
```

