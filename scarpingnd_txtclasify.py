# Web Scraping + Text Classification in Python
# Complete guide with practical examples

# ============================================================================
# PART 1: WEB SCRAPING
# ============================================================================

# Required libraries:
# pip install requests beautifulsoup4 selenium pandas scikit-learn

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from urllib.parse import urljoin

# --- Example 1: Basic Web Scraping with BeautifulSoup ---
def scrape_basic_page(url):
    """Scrape text content from a webpage"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract different elements
        title = soup.find('title').text if soup.find('title') else 'No title'
        
        # Get all paragraphs
        paragraphs = [p.text.strip() for p in soup.find_all('p')]
        
        # Get all links
        links = [a.get('href') for a in soup.find_all('a', href=True)]
        
        return {
            'url': url,
            'title': title,
            'paragraphs': paragraphs,
            'links': links,
            'text': ' '.join(paragraphs)
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

# --- Example 2: Scraping Multiple Pages ---
def scrape_news_articles(base_url, num_articles=10):
    """Scrape multiple news articles"""
    articles = []
    
    # Get main page
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find article links (adjust selector based on website structure)
    article_links = soup.find_all('a', class_='article-link')[:num_articles]
    
    for link in article_links:
        article_url = urljoin(base_url, link.get('href'))
        article_data = scrape_basic_page(article_url)
        
        if article_data:
            articles.append(article_data)
    
    return pd.DataFrame(articles)

# --- Example 3: Advanced Scraping with Headers ---
def scrape_with_headers(url):
    """Scrape with custom headers to avoid blocking"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(['script', 'style', 'nav', 'footer']):
        script.decompose()
    
    text = soup.get_text(separator=' ', strip=True)
    return text

# ============================================================================
# PART 2: TEXT CLASSIFICATION
# ============================================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# --- Example 4: Text Preprocessing ---
def preprocess_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# --- Example 5: Training a Text Classifier ---
def train_text_classifier(texts, labels):
    """Train a text classification model"""
    
    # Preprocess texts
    texts = [preprocess_text(t) for t in texts]
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=1,  # Changed from 2 to 1 for smaller datasets
        stop_words='english'
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train classifier
    classifier = MultinomialNB()
    classifier.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    print(f"Accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(report)
    
    return vectorizer, classifier

# --- Example 6: Sentiment Analysis Classifier ---
def classify_sentiment(text, vectorizer, classifier):
    """Classify text sentiment"""
    processed = preprocess_text(text)
    vectorized = vectorizer.transform([processed])
    prediction = classifier.predict(vectorized)[0]
    probabilities = classifier.predict_proba(vectorized)[0]
    
    return {
        'text': text,
        'sentiment': prediction,
        'confidence': max(probabilities)
    }

# ============================================================================
# PART 3: COMPLETE PIPELINE (Scraping + Classification)
# ============================================================================

def scrape_and_classify_pipeline(urls, vectorizer, classifier):
    """Complete pipeline: scrape websites and classify content"""
    results = []
    
    for url in urls:
        print(f"Processing: {url}")
        
        # Scrape
        text = scrape_with_headers(url)
        
        if text:
            # Classify
            prediction = classify_sentiment(text, vectorizer, classifier)
            prediction['url'] = url
            results.append(prediction)
    
    return pd.DataFrame(results)

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Basic scraping
    print("=== Example 1: Basic Scraping ===")
    # data = scrape_basic_page("https://example.com")
    
    # Example 2: Training a classifier with sample data
    print("\n=== Example 2: Training Classifier ===")
    
    # Sample training data (expanded dataset)
    sample_texts = [
        # Positive samples
        "This product is amazing! I love it so much.",
        "Pretty good, meets my expectations.",
        "Best purchase ever! Highly satisfied.",
        "Excellent quality and fast shipping.",
        "Outstanding service and great value for money.",
        "Very happy with this purchase, highly recommend.",
        "Exceeded my expectations, will buy again.",
        "Fantastic experience from start to finish.",
        "Superb quality, exactly what I needed.",
        "Incredible product, worth every penny.",
        "Love it! Perfect for my needs.",
        "Great product, fast delivery, very satisfied.",
        "Impressive quality and excellent customer service.",
        "Wonderful purchase, couldn't be happier.",
        "Brilliant product, works perfectly.",
        "Amazing value, highly recommend to everyone.",
        "Perfect! Just what I was looking for.",
        "Exceptional quality and great price.",
        "Very pleased with this purchase.",
        "Excellent product, fast shipping, great seller.",
        # Negative samples
        "Terrible experience, would not recommend.",
        "Absolutely horrible, waste of money.",
        "Not what I expected, quite disappointing.",
        "Poor customer service and low quality.",
        "Worst purchase I've ever made.",
        "Complete waste of money, doesn't work.",
        "Very disappointed, poor quality.",
        "Awful product, broke after one use.",
        "Do not buy this, total ripoff.",
        "Horrible experience, slow shipping.",
        "Terrible quality, not worth the price.",
        "Very unhappy with this purchase.",
        "Poor quality control, arrived damaged.",
        "Disappointing product, doesn't work as advertised.",
        "Bad experience, would not recommend.",
        "Cheaply made, fell apart quickly.",
        "Not satisfied at all, returning it.",
        "Waste of time and money.",
        "Poor design and terrible quality.",
        "Very frustrating experience overall.",
    ]
    
    sample_labels = [
        'positive', 'positive', 'positive', 'positive', 'positive',
        'positive', 'positive', 'positive', 'positive', 'positive',
        'positive', 'positive', 'positive', 'positive', 'positive',
        'positive', 'positive', 'positive', 'positive', 'positive',
        'negative', 'negative', 'negative', 'negative', 'negative',
        'negative', 'negative', 'negative', 'negative', 'negative',
        'negative', 'negative', 'negative', 'negative', 'negative',
        'negative', 'negative', 'negative', 'negative', 'negative',
    ]
    
    # Train model
    vectorizer, classifier = train_text_classifier(sample_texts, sample_labels)
    
    # Example 3: Classify new text
    print("\n=== Example 3: Classify New Text ===")
    test_text = "This is a fantastic product, I'm very happy with it!"
    result = classify_sentiment(test_text, vectorizer, classifier)
    print(f"Text: {result['text']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.3f}")
    
    # Example 4: Full pipeline
    print("\n=== Example 4: Full Pipeline ===")
    print("Ready to scrape and classify URLs!")
    # urls = ["https://example1.com", "https://example2.com"]
    # results_df = scrape_and_classify_pipeline(urls, vectorizer, classifier)
    # print(results_df)