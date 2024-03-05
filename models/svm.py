import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

file_path= "C:/Users/kakis/Downloads/train_emotion.csv"
data= pd.read_csv(file_path)

# We split the dataset into training and validation sets
train_data, validation_data, train_labels, validation_labels = train_test_split(
    data['text'], data['emotion'], test_size=0.2, random_state=42)

# Setting up a pipeline with TF-IDF and SVM
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', SVC())
])

# Parameters for tuning
param_grid = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],  # Unigrams and bigrams
    'classifier__C': [1, 10],  # Regularization strength
    'classifier__kernel': ['linear', 'rbf']  # Kernel type
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
print("Best Parameters: ", best_params)
print("Best Score: ", best_score)
print(report)


