# HARE: Highlighting Annotator for Ranking and Exploration

This open-source software package implements two components of a pipeline for identifying information relevant to a specific topic in text documents:

1. **Machine-learning backend**, for training and testing a machine-learning based neural network model that assigns a relevance score to each token in a document;
2. **Front-end web interface**, for viewing relevance tags produced by information extraction models, ranking documents by relevance, and analyzing qualitative model outcomes.

This system was described in the following paper:

+ D Newman-Griffis and E Fosler-Lussier, "[HARE: a Flexible Highlighting Annotator for Ranking and Exploration](https://www.aclweb.org/anthology/D19-3015)". In _Proceedings of EMNLP 2019_.

## Setup/Installation

The included `makefile` provides pre-written commands for common tasks in the HARE backend.

The `requirements.txt` file lists all required Python packages installable with pip. Just run
```
pip install -r requirements.txt
```
to install all packages.

Source code of two packages is required for generating BERT features; these packages ([BERT](https://github.com/google-research/bert) and [bert\_to\_hdf5](https://github.com/drgriffis/bert_to_hdf5)) are automatically downloaded by the `utils/get_bert_hdf5_features.sh` script.

## Package components

The processing pipeline in this package includes several primary elements, described here with reference to key code files. (For technical reference on script usage, see `makefile`)

- **Text preprocessing:** tokenization and formatting for analysis.
  + See ```data/extract_data_files.py```
- **Contextualized feature extraction:** pre-calculation of embedding features, using contextualized language models.
  + If using static embeddings, features are extracted dynamically at runtime.
  + See ```utils/get_bert_hdf5_features.sh``` and ```makefile```
- **Cross-validation split generation:** pre-generation of cross-validation splits for a specified dataset, for consistency across experiments.
  + Splitting is done at the document level (assumes documents are independent).
  + See ```experiments.document_splits```
- **Token relevance model training:** implementation and training of token-level relevance estimator.
  + Implemented in TensorFlow.
  + For model, see ```model/DNNTokenClassifier.py```
  + For training, see ```experiments/train.py```
- **Prediction with token relevance model:** application of pre-trained token-level relevance estimator to new data.
  + See ```experiments/run_pretrained.py```
- **Document-level results visualization:** web-based viewing of token-level relevance predictions.
  + See <a href="visualization/README.md">visualization README</a> for more details.
- **Document ranking:** web-based interface ranking documents by relevance scores.
  + See <a href="visualization/README.md">visualization README</a> for more details.
- **Qualitative outcomes analysis:** web-based interface for analyzing qualitative trends in model outputs.
  + See <a href="visualization/README.md">visualization README</a> for more details.

## Demo script/data

This package includes two tiny datasets for code demonstration purposes:

- ```demo_data/demo_labeled_dataset``` 5 short, synthetic clinical documents with mobility-related information. Text files are located in the `txt` subdirectory, and `csv` contains corresponding CSV files with standoff annotations.
- ```demo_data/demo_unlabeled_dataset``` 5 more short, synthetic clinical documents, only one of which contains mobility-related information. Text files are provided without corresponding annotations.

The included `run_demo_experiments.sh` script (written for Bash execution on a Unix machine) utilizes the provided make targets to run a complete end-to-end experiment, with the following steps:

1. Tokenize both labeled and unlabeled datasets, using SpaCy and WordPiece.
2. Generate contextualized embedding features for each dataset, using ELMo and BERT.
3. Train HARE models on the labeled dataset, using static embeddings, ELMo, and BERT.
4. Use the trained models to get predictions on the unlabeled dataset.
5. Prepare all output predictions for viewing in the front-end interface.

## Reference

If you use this software in your own work, please cite the following paper:
```
@inproceedings{newman-griffis-fosler-lussier-2019-hare,
  title = "{HARE}: a Flexible Highlighting Annotator for Ranking and Exploration",
  author = "Newman-Griffis, Denis and Fosler-Lussier, Eric",
  booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP): System Demonstrations",
  month = nov,
  year = "2019",
  address = "Hong Kong, China",
  publisher = "Association for Computational Linguistics",
  url = "https://www.aclweb.org/anthology/D19-3015",
  doi = "10.18653/v1/D19-3015",
  pages = "85--90",
}
```

## License

All source code, documentation, and data contained in this package are distributed under the terms in the LICENSE file (modified BSD).
