# imdb-sentiment-analysis
IMDb movie review sentiment classification using TF-IDF, Logistic Regression, and Naive Bayes.
# IMDb Movie Review Sentiment Analysis

## Project Overview

This project uses Natural Language Processing (NLP) and machine learning to classify IMDb movie reviews as **positive** or **negative**. It compares two classification algorithms—Logistic Regression and Multinomial Naive Bayes—using TF-IDF text features.

This project was completed as part of **Elevvo Pathways — NLP Internship, Task 1: Sentiment Analysis**.

## Objectives

- Load and explore a movie-review dataset.
- Clean and preprocess review text.
- Convert text into numerical features using TF-IDF.
- Train Logistic Regression and Multinomial Naive Bayes classifiers.
- Evaluate model performance using standard classification metrics.
- Visualize frequent words in positive and negative reviews.
- Test the trained model on custom review examples.

## Dataset

The project uses an IMDb movie-review dataset with the following characteristics:

| Property | Details |
|---|---|
| Total reviews | 50,000 |
| Positive reviews | 25,000 |
| Negative reviews | 25,000 |
| Input text column | `review` |
| Sentiment column | `sentiment` |
| Classification task | Binary sentiment classification |

The dataset contains balanced positive and negative classes. The CSV file used in the notebook is named `IMDB.csv`.

**Dataset note:** The dataset file is not included in this repository by default. Obtain it from the source you used, ensure its usage terms permit your intended use, and upload it to the notebook environment when prompted.

## Technologies Used

- Python
- pandas and NumPy
- Matplotlib
- scikit-learn
- Regular expressions and Python's `html` module
- Google Colab
- Git and GitHub

## Methodology

### 1. Data loading and exploration

Loaded the CSV dataset and examined its dimensions, columns, missing values, and sentiment distribution.

### 2. Text preprocessing

The review-cleaning function:
- Decodes HTML entities.
- Removes HTML tags.
- Converts text to lowercase.
- Normalizes repeated whitespace.

Stopword removal is not applied to the main model input, helping preserve potentially important sentiment expressions such as negation.

### 3. Train-test split

The dataset was split into:
- **80% training data:** 40,000 reviews.
- **20% test data:** 10,000 reviews.

The split used `random_state=42` and stratification to preserve the class distribution.

### 4. TF-IDF feature extraction

`TfidfVectorizer` converted review text into numerical features using:
- Maximum features: 20,000
- N-gram range: unigrams and bigrams `(1, 2)`
- Minimum document frequency: 2
- Sublinear term-frequency scaling enabled

The vectorizer was fitted on the training reviews and then applied to the test reviews.

### 5. Model training

Two classifiers were trained using the same TF-IDF features:

1. **Logistic Regression**
2. **Multinomial Naive Bayes**

### 6. Evaluation

Models were compared using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices

The project also includes word-frequency visualizations and predictions for custom review examples.

## Results

The following metrics were obtained from the held-out test set.

| Metric | Logistic Regression | Multinomial Naive Bayes |
|---|---:|---:|
| Accuracy | 90.80% | 87.83% |
| Precision | 90.36% | 87.15% |
| Recall | 91.34% | 88.74% |
| F1-score | 90.85% | 87.94% |

### Interpretation

Logistic Regression achieved the highest score across all four reported metrics in this experiment. Its accuracy was **2.97 percentage points higher** than that of Multinomial Naive Bayes.

These results indicate that Logistic Regression performed better on this test split. They do not guarantee the same performance on every dataset or on real-world reviews.

## Visualizations

The notebook includes:
- Sentiment class distribution.
- Comparison of model evaluation metrics.
- Confusion matrices for both classifiers.
- Frequent-word charts for positive and negative reviews.

## Custom Review Testing

The trained Logistic Regression model was also tested with example reviews that were not part of the training dataset. The notebook displays the predicted sentiment and the model's estimated positive-class probability for each example.

These demonstrations illustrate model usage; individual predictions may be incorrect.

## Repository Structure

```text
imdb-sentiment-analysis/
├── sentiment_analysis.ipynb
├── model_comparison_results.csv
└── README.md
```

- `sentiment_analysis.ipynb` — notebook containing the data preparation, training, evaluation, visualizations, and custom predictions.
- `model_comparison_results.csv` — exported model comparison metrics.
- `README.md` — project documentation.

## How to Run

1. Open the notebook in Google Colab.
2. Upload the `IMDB.csv` dataset when prompted.
3. Run the notebook cells in order, from data loading through evaluation.
4. Review the evaluation metrics and generated visualizations.
5. Run the custom-review cell to test example predictions.

The notebook was developed for Google Colab. If you run it in another environment, install the required libraries and adapt the dataset-loading cell as necessary.

## Limitations

- The experiment uses a single train-test split.
- The models may struggle with sarcasm, context-dependent language, or subtle sentiment.
- Frequent words are not necessarily the most predictive words.
- The reported scores measure performance on this dataset's held-out test set and may not transfer directly to other review sources.

## Future Improvements

- Use cross-validation for more robust model comparison.
- Tune model hyperparameters.
- Examine misclassified reviews.
- Test additional models and feature representations.
- Evaluate on a separate dataset to assess generalization.

## Author

**Stewart Daniel**

Machine Learning and Natural Language Processing project.

## Acknowledgements

Completed as part of the Elevvo Pathways NLP Internship task on sentiment analysis.
