# HARE Front-end interface

Flask-based interface for reviewing and analyzing relevance predictions.  The interface includes three components:

- <a href="#document-ranking">Document ranking</a>
- <a href="#document-viewing">Document viewing</a>
- <a href="#qualitative-analysis">Qualitative analysis</a>

These components are detailed below, along with a <a href="#annotation-sets-data-models">description of the data models used</a>.

## Annotation sets (data models)

The output produced by running a relevance-tagging model (e.g., token-level HARE tagger, NER models) on any set of documents is referred to as an **annotation set.**

Annotation sets are configured using a tab-separated file, configured by the ```FileMap``` setting in the ```Predictions``` section of ```viz_config.ini```.

The annotation set mapping file should be formatted as:
```
<Name> \t <Labeled|Unlabeled> \t <Absolute path to output file>
```

An annotation set can be **labeled** if gold labels are provided for each token, or **unlabeled** if it is the result of running a model on new data.
- For examples of labeled and unlabeled model output files, run the `run_demo_experiments.sh` script and look in ```demo_data/demo_outputs```.

Clicking on any annotation set in the interface will take you to the Document Ranking interface.

## Document ranking

The document ranking interface lists all documents contained in an annotation set, along with annotation set-level statistics.

At any point, a different annotation set can be selected to view outcomes from those data.

The interface contains several components:

### List of documents

Documents in the selected annotation set are listed in tabular format, with indication of:
- Name of the document: clicking this button will take you to the Document Viewing interface for the selected document.
- Ranking assigned by the model
- Score assigned by the model (see description of document scorers below)
- Ranking assigned by gold labels (if available)
- Score assigned by gold labels (if available)

### Ranking Statistics panel

This panel presents the following statistics and information:

- Number of documents in the dataset
- Spearman's rank correlation coefficient between model ranking and gold ranking (if available)
- Selector for scoring model to use for gold labels (if available)
- Selector for scoring model to use for model outputs
- Checkbox to show documents ranked by gold label scores (if available) or model scores (default)
- Checkbox to show visualized classes using gold label scores (if available) or model scores (default)
- Binarization threshold to convert continuous relevance scores into Relevant/Not relevant categories (used by CountSegmentsAndTokens scorer)
- Number of blanks (Not relevant tokens) to allow between Relevant tokens when calculating contiguous relevant segments
- Checkbox to use Viterbi smoothing for relevance scores or not

### Visualized class configuration

For visual analysis, document scores can be assigned visual classes using configurable class bins based on assigned relevance score.

A table listing the current class configuration, and corresponding score bins, is shown below the Ranking Statistics.

Class configuration is set in `viz_config.ini`, using the ```Classification``` setting.

### Annotation Evaluation Metrics (if data are labeled)

Displays evaluation metrics calculated for the full annotation set, at the token level.

### Model Annotation Statistics

Displays summary statistics of patterns in relevant segments predicted by model outputs.

### Gold Annotation Statistics (if data are labeled)

Displays summary statistics of patterns in relevant segments according to gold labels.

### Scoring models

Three different scoring models are implemented:

- **CountSegmentsAndTokens:** document score is assigned as the number of relevant segments (contiguous sequences of relevant tokens).
  + If two documents have the same number of segments, the one with more relevant tokens is ranked higher.
  + Note that this scorer converts relevance scores to binary Relevant/Not relevant labels, using the configurable binarization threshold.
- **SumTokenScores:** document score is the sum of token-level relevance scores (not binarized).
- **DensityScorer:** document score is the sum of token-level relevance scores, normalized by document length.

## Document viewing

The document viewing interface displays token-level relevance information from the current annotation set.

If another annotation set contains a document of the same name, that annotation set can be selected from the drop-down at the top of the interface to compare outputs.

### Document text view

All tokens in the document are displayed, with the following markup:
- If a token is relevant according to the current binarization threshold, it is <span style="background-color:yellow;">highlighted in yellow</span>.
- If the data are labeled and the token is marked as Relevant in gold labels, it is shown in <span style="color:red; font-weight:bold;">bold red font</span>.

Any token can be clicked on to display the relevance score assigned to it.

### Labeling Settings panel

This panel allows configuration of the visualization settings:
- Binarization threshold for converting continuous relevance scores to Relevant/Not relevant labels
- Number of blanks (not relevant tokens) allowed between relevant tokens when identifying relevant segments
- Checkbox for Viterbi smoothing correction of token-level relevance scores

### Document Statistics

Summary statistics of model predictions on this specific document.
- Number of segments marked as relevant by the model
- Number of segments marked as relevant by gold labels (if available)
- Number of tokens marked relevant by the model
- Number of tokens relevant according to gold labels (if available)
- Accuracy of binarized token-level predictions (if labeled)
- Precision of binarized token-level predictions (if labeled)
- Recall of binarized token-level predictions (if labeled)
- F-1 and F-2 of binarized token-level predictions (if labeled)

### Relevance score distribution

Histogram showing distribution of token-level relevance scores produced by the model for this document.

### Thresholding distribution (if data are labeled)

Line plots showing how precision, recall, and F-measure vary over all possible binarization thresholds for this document.

Visualization includes the threshold yielding highest F-measure.

## Qualitative analysis

This interface displays a few different qualitative analyses of model outcomes within an annotation set.

### Lexicalization

Table displaying mean relevance scores per token type in the annotation set.
- If data are labeled, model lexicalization is compared to lexicalization measured from gold labels, and rank correlation coefficient is given at the head of the table
- Ranking is displayed using model relevance and gold labels (if available)
- Lexicalization is only measured for types above a minimum frequency in the annotation set; this threshold is configurable.

### Relevance score distribution

Histogram showing distribution of token-level relevance scores produced by the model for the full annotation set.

### Thresholding distribution (if data are labeled)

Line plots showing how precision, recall, and F-measure vary over all possible binarization thresholds for the full annotation set.

### Annotation evaluation metrics (if data are labeled)

Displays evaluation metrics calculated for the full annotation set, at the token level.

### Model annotation statistics

Displays summary statistics of patterns in relevant segments predicted by model outputs.

### Gold annotation statistics (if data are labeled)

Displays summary statistics of patterns in relevant segments according to gold labels.

## License

All source code, documentation, and data contained in this package are distributed under the terms in the LICENSE file (modified BSD).
