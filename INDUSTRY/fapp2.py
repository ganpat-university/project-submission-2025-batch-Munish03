import torch
import torch.nn as nn
import mysql.connector
import pickle
import spacy
from textblob import TextBlob
from flask import Flask, request, render_template
from transformers import BertTokenizer

# Load the model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

class SentimentLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super(SentimentLSTM, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out[:, -1, :])

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Fix tokenizer vocab size issue
vocab_size = tokenizer.vocab_size

# Load best model
model = SentimentLSTM(vocab_size, 128, 256, 3)
model.load_state_dict(torch.load("best_sentiment_model.pth", map_location=device))
model.to(device)
model.eval()
print("✅ Best model loaded successfully.")

# Load NLP model
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

# Database config
db_config = {
    'host': "localhost",
    'user': 'Munish',
    'password': 'Munish03',
    'database': 'hotel_reviews',
}

def extract_keywords(text):
    """Extracts keywords from the text using spaCy."""
    doc = nlp(text)
    keywords = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN", "ADJ"]]
    return ", ".join(keywords)

def extract_sentiment_words(text):
    """Extracts sentiment words using TextBlob."""
    blob = TextBlob(text)
    sentiment_words = [word for word, tag in blob.tags if tag in ['JJ', 'RB', 'VB']]
    return ", ".join(sentiment_words)

def save_to_database(review_text, sentiment, keywords, sentiment_words):
    """Save review, sentiment, keywords, and sentiment words to MySQL database."""
    try:
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        query = """
        INSERT INTO reviews (review_text, sentiment, keywords, sentiment_words) 
        VALUES (%s, %s, %s, %s)
        """
        cursor.execute(query, (review_text, sentiment, keywords, sentiment_words))
        connection.commit()
        cursor.close()
        connection.close()
        print("✅ Data saved to database successfully!")
    except mysql.connector.Error as err:
        print(f"❌ Database Error: {err}")

def analyze_text(text):
    """Analyze sentiment of the given text using the trained model."""
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    
    with torch.no_grad():
        output = model(input_ids)
        sentiment = torch.argmax(output, dim=1).item()
    
    return ["Negative", "Neutral", "Positive"][sentiment]

@app.route("/", methods=["GET", "POST"])
def analyze_sentiment():
    if request.method == "POST":
        review_text = request.form["review_text"]
        sentiment = analyze_text(review_text)
        keywords = extract_keywords(review_text)
        sentiment_words = extract_sentiment_words(review_text)
        save_to_database(review_text, sentiment, keywords, sentiment_words)
        return render_template("result.html", review_text=review_text, sentiment=sentiment, keywords=keywords, sentiment_words=sentiment_words)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
