import os
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from imblearn.over_sampling import SMOTE  # Added SMOTE for handling imbalance
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# Load vocabulary and sentiment ratings
def load_vocab_and_ratings(vocab_file, ratings_file):
    with open(vocab_file, 'r', encoding='utf-8', errors='ignore') as vf:
        vocab = [line.strip() for line in vf]
    with open(ratings_file, 'r', encoding='utf-8', errors='ignore') as rf:
        ratings = [float(line.strip()) for line in rf]
    return dict(zip(vocab, ratings))

# GUI Configuration
def run_analysis():
    # Collect inputs from GUI
    folder_path = folder_entry.get()
    c_value = float(c_entry.get())
    threshold = float(threshold_entry.get())
    sentiment = float(sentiment_meter_entry.get())
    use_smote = smote_var.get()

    if not os.path.exists(folder_path):
        messagebox.showerror("Error", "Invalid folder path.")
        return

    try:
        # Progress bar setup
        progress_bar['value'] = 0
        root.update_idletasks()

        # Load vocabulary and ratings
        sentiment_dict = load_vocab_and_ratings('imdb.vocab', 'imdbEr.txt')

        # Determine file names
        filenames = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

        # Display ratio in results
        results_text.delete('1.0', tk.END)  # Clears previous results

        # Text preprocessing function
        def preprocess_text(text):
            text = text.lower()
            text = re.sub(f"[{string.punctuation}]", "", text)
            text = re.sub(r"\d+", "", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        # Calculate review sentiment based on word ratings
        def calculate_sentiment_score(text):
            words = text.split()
            scores = [sentiment_dict.get(word, 0) for word in words]
            avg_score = np.mean(scores) if scores else 0
            return avg_score

        # Sentiment analysis function using both VADER and custom dictionary
        def get_sentiment(text):
            analyzer = SentimentIntensityAnalyzer()
            vader_score = analyzer.polarity_scores(text)['compound']
            custom_score = calculate_sentiment_score(text)
            final_score = (vader_score + custom_score) / 2
            if final_score >= sentiment/10:
                return 1  # Positive
            elif final_score <= -sentiment/10:
                return 0  # Negative
            else:
                return 2  # Neutral

        # Load Dataset
        reviews, sentiments = [], []
        for i, filename in enumerate(filenames):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8', errors='ignore') as file:
                review = file.read().strip()
                if review:
                    reviews.append(review)
                    sentiments.append(get_sentiment(review))
            progress_bar['value'] = (i + 1) / len(filenames) * 20
            root.update_idletasks()

        # Preprocess reviews
        data = [preprocess_text(review) for review in reviews]
        labels = sentiments
        data = [review for review in data if len(review.strip()) > 0]
        labels = [labels[i] for i in range(len(labels)) if len(data[i].strip()) > 0]

        # Vectorization
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        X = vectorizer.fit_transform(data)
        y = np.array(labels)
        progress_bar['value'] = 40
        root.update_idletasks()

        # Apply SMOTE if selected
        if use_smote:
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
        progress_bar['value'] = 60
        root.update_idletasks()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Training
        model = LogisticRegression(C=c_value, solver='liblinear', max_iter=1000)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        progress_bar['value'] = 80
        root.update_idletasks()

        # Apply threshold
        y_pred = (y_prob >= threshold).astype(int)

        # Evaluation
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Sentiment Meter
        total_reviews = len(y_test)
        positive_count = sum(y_pred)
        negative_count = total_reviews - positive_count
        sentiment_meter = f"Positive: {positive_count}/{total_reviews} ({(positive_count/total_reviews)*100:.2f}%)\n"
        sentiment_meter += f"Negative: {negative_count}/{total_reviews} ({(negative_count/total_reviews)*100:.2f}%)\n"

        # Display results
        results_text.insert(tk.END, f"Accuracy: {acc}\n")
        results_text.insert(tk.END, sentiment_meter)
        results_text.insert(tk.END, f"Classification Report:\n{report}\n")
        results_text.insert(tk.END, f"Confusion Matrix:\n{cm}\n")

        # Visualization
        progress_bar['value'] = 100

    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI Setup
root = tk.Tk()
root.title("Sentiment Analysis Tool")
root.geometry("600x600")

# Folder Selection
tk.Label(root, text="Dataset Folder:").pack()
folder_entry = tk.Entry(root, width=50)
folder_entry.pack()
tk.Button(root, text="Browse", command=lambda: folder_entry.insert(0, filedialog.askdirectory())).pack()

# Hyperparameters
tk.Label(root, text="Logistic Regression:").pack()
c_entry = tk.Entry(root, width=10)
c_entry.insert(0, "100")
c_entry.pack()

tk.Label(root, text="Cut-off Value (0.0 - 1.0):").pack()
threshold_entry = tk.Entry(root, width=10)
threshold_entry.insert(0, "0.5")
threshold_entry.pack()

tk.Label(root, text="Sentiment Meter (0.0 - 1.0):").pack()
sentiment_meter_entry = tk.Entry(root, width=10)
sentiment_meter_entry.insert(0, "0.5")
sentiment_meter_entry.pack()

# SMOTE Option
tk.Label(root, text="Use SMOTE (Results Balancing):").pack()
smote_var = tk.BooleanVar()
smote_check = tk.Checkbutton(root, text="Enable SMOTE", variable=smote_var)
smote_check.pack()

# Progress Bar
progress_bar = ttk.Progressbar(root, length=500, mode='determinate')
progress_bar.pack(pady=10)

# Run Analysis Button
tk.Button(root, text="Run Analysis", command=run_analysis).pack()

# Results Display
results_text = tk.Text(root, height=20, width=70)
results_text.pack()

root.mainloop()