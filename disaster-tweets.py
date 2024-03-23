import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Step 1: Data Exploration
train_data = pd.read_csv("./data/train.csv")

# Step 2: Data Preprocessing
X = train_data['text']
y = train_data['target']

# Step 3: Model Training
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', RandomForestClassifier())
])

parameters = {
    'tfidf__max_df': (0.5, 0.75, 1.0),
    'clf__n_estimators': (50, 100, 200),
    'clf__max_features': ['sqrt'],  # Wrap 'sqrt' in a list
    'clf__max_depth' : (None, 50, 100),
    'clf__min_samples_split': (2, 5, 10),
    'clf__min_samples_leaf': (1, 2, 4)
}

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

random_search = RandomizedSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1, n_iter=50)
random_search.fit(X_train, y_train)

pkl_filename = "disaster-tweets.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(random_search, file)


# Step 4: Model Evaluation
y_pred = random_search.predict(X_valid)
f1 = f1_score(y_valid, y_pred)
print("F1 Score:", f1)

# Step 5: Making Predictions
test_data = pd.read_csv("./data/test.csv")
X_test = test_data['text']
predictions = random_search.predict(X_test)

# Step 6: Creating Submission File
submission = pd.DataFrame({'id': test_data['id'], 'target': predictions})
submission.to_csv('submission.csv', index=False)