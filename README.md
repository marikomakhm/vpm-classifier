# VPM Web Page Classifier

The VPM web page classifier determines the presence of virtual patent marking (VPM) information in a web page.

#### How does it work?

The classifier uses the web page URL and HTML to extract features, to predict whether the web page should belong to one of the following categories:

- ***Simple VPM page (SVPM):*** the primary purpose of the web page is to present patent information about products.
- ***Hidden VPM page (HVPM):*** the web page contains VPM information, but it is not the primary purpose of the page (e.g. product catalogues, e-commerce pages, product user guides).
- ***Non-VPM page (NVPM):*** the web page contains no VPM information.

In this project, one is able to build and train the classifier, as well as use it to predict the category of previously unseen web page data. The prediction function outputs a file containing, for each sample, the probability that the sample belongs to each of the categories described above.

We invite the reader to read `report.pdf` for further information on how the classifier works.

## Getting started

### Prerequisites

To use this project, you must have Python 3.7 installed.

You must also have the following packages installed: NumPy, Pandas, scikit-learn, spaCy, Seaborn, Matplotlib, NLTK, Gensim, html2text, LangID, Beautiful Soup, LightGBM. You must also install the following spaCy model: `en_core_web_md`.

You can install the package and model prerequisites by running the following command:

```shell
pip install numpy pandas scikit-learn lightgbm spacy seaborn matplotlib nltk gensim html2text langid beautifulsoup4
python -m spacy download en_core_web_md
```

In order to train the model (i.e. to run `train.py`), you must prepare a data records CSV file that contains, for each training sample, tuples of the following format: `(filename, url, cat)`, where `filename` is the absolute path to the sample HTML file, `url` is its corresponding URL, and `cat` is the assigned category of the sample.

In order to use the model for prediction (i.e. to run `train.py`), you must prepare a data prediction CSV file that contains, for each prediction sample, tuples of the following format: `(filename, url)`, where `filename` is the absolute path to the sample HTML file, and `url` is its corresponding URL.

### Usage

#### Training

To train a model on existing data, run the following command:

```shell
python train.py [--s path_to_source_dir] --d path_to_data_records
```

*Note 1: training can take up to 30 minutes based on experiments.*

*Note 2: if you wish to change model parameters, you can do so by manually changing the parameters in `train.py`.*

Running `train.py` produces the following output files in the source directory (paths are relative to the source directory):

| Directory                                      | Description                                                  |
| ---------------------------------------------- | ------------------------------------------------------------ |
| `output/data/`                                 | Contains text extracted from HTML files in data records file. |
| `output/models/feature_model.sav`              | Trained feature model.                                       |
| `output/models/best_features.txt`              | Contains list of best input features extracted using random forest estimator cross validation for the feature model. |
| `output/models/patent_number_vector_model.sav` | Trained patent number vector model.                          |
| `output/models/patent_vector_model.sav`        | Trained patent vector model.                                 |
| `output/models/stacked_model.sav`              | Trained stacked model.                                       |
| `output/train_features_df.csv`                 | Stores extracted features for files in data records (input features for feature model, and extracted text from HTML). |
| `errors/preprocessing.csv`                     | If any errors are encountered in the preprocessing stage (regarding text extraction from HTML), this file contains the file names and their corresponding error codes (the error code significations can be found in `preprocessing.py`). |

For details on the model variants, we invite the reader to read `report.pdf`.

#### Prediction

To use the trained model to predict the category of a set of samples, run the following command:

```shell
python predict.py [--s path_to_source_dir] --d path_to_data_predict --m path_to_trained_models
```

Running `predict.py` produces the following output files in the source directory (paths are relative to the source directory):

| Directory                        | Description                                                  |
| -------------------------------- | ------------------------------------------------------------ |
| `output/data/`                   | Contains text extracted from HTML files in data prediction file. |
| `output/predict_features_df.csv` | Stores extracted features for files in data predictions file (input features for feature model, and extracted text from HTML). |
| `output/predictions.csv`         | Rows of the following format: `(filename, pred_hvpm, pred_nvpm, pred_svpm, pred)`, representing the probability that each file belongs in each category based on the stacked model, as well as the final predicted category. |
| `errors/preprocessing.csv`       | If any errors are encountered in the preprocessing stage (regarding text extraction from HTML), this file contains the file names and their corresponding error codes (the error code significations can be found in `preprocessing.py`). |

## Structure

The table below describes the files in the project:

| Directory                              | Description                                                  |
| -------------------------------------- | ------------------------------------------------------------ |
| `train.py`                             | Processes input data, extracts features and trains classification model. |
| `predict.py`                           | Predicts categories of input data using trained model.       |
| `preprocessing.py`                     | Contains functions to preprocess data, such as extracting text from HTML. |
| `feature_processing.py`                | Contains functions to extract HTML and URL based statistical features. |
| `vector_processing.py`                 | Contains functions to build vector representations of samples using regular expression matching. |
| `model.py`                             | Contains functions to train, test and save feature-input, vector-input and aggregate input models. |
| `performance.py`                       | Contains functions that display model performance to the user. |
| `notebooks/01_data_stats.ipynb`        | Notebook containing initial statistical analysis of data (first notebook to read/run). |
| `notebooks/02_baseline_model.ipynb`    | Notebook containing URL baseline model results (second notebook to read/run). |
| `notebooks/03_footer_extraction.ipynb` | Notebook containing footer extraction justification process (third notebook to read/run). |
| `notebooks/helpers.py`                 | Helper functions for the notebooks (mainly similar functions to those found in `feature_processing.py` and `performance.py`) |
| `notebooks/sample_data.csv`            | Statistical feature data, to reproduce notebook data if needed (from actual data used for this project). |
| `report.pdf`                           | A report documenting the details of this project, and explaining the theoretical reasoning behind the design decisions. |

## Authors and acknowledgements

This project was authored by Mariko Makhmutova, as the optional semester project towards the Master's program in Data Science at EPFL, Spring 2020.

The author would like to thank Professor Gaétan de Rassenfosse and Samuel Arnod-Prin for their guidance and help throughout this project. The author would also like to acknowledge Albert Calvo for his efforts in building a similar virtual patent marking classifier, as components of his work served as a starting point for this project.