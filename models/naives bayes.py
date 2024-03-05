import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

file_path = "C:/Users/kakis/Downloads/train_emotion.csv"
data = pd.read_csv(file_path)

# Splitting the dataset into training and validation sets
train_data, validation_data, train_labels, validation_labels = train_test_split(
    data['text'], data['emotion'], test_size=0.2, random_state=42)

# Setting up a pipeline with TF-IDF and Multinomial Naive Bayes
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', MultinomialNB())
])

# Reduced Parameters for tuning
param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],  # Unigrams and bigrams
    'tfidf__max_df': [0.5, 0.75, 1.0],  # Maximum document frequency
    'tfidf__min_df': [1, 2, 3],  # Minimum document frequency
    'classifier__alpha': [0.1, 1, 10]  # Additive (Laplace/Lidstone) smoothing parameter
}

# Grid search with 3-fold cross-validation
grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(train_data, train_labels)

# Best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Evaluate the best model on the validation data
best_model = grid_search.best_estimator_
validation_predictions = best_model.predict(validation_data)
report = classification_report(validation_labels, validation_predictions)

# Print best parameters, best score, and classification report
print("Best Parameters: ", best_params)
print("Best Score: ", best_score)
print(report)
