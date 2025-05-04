import pandas as pd
import kagglehub
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Download the Kaggle dataset
path = kagglehub.dataset_download("andrewmvd/trip-advisor-hotel-reviews")

# Load the dataset
df = pd.read_csv(path + "/tripadvisor_hotel_reviews.csv")
df.columns = ["review_text", "star_rating"]
df = df.dropna()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(df["review_text"], df["star_rating"], test_size=0.2, random_state=42)

# Vectorize the text
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Train the model using Grid Search
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(X_train_vectors, y_train)

# Best model
model = grid.best_estimator_

# Evaluate the model
y_pred = model.predict(X_test_vectors)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
with open("sentiment_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully.")
