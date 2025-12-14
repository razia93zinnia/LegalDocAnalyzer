from collections import Counter
import re
from flask import Flask, request, render_template, jsonify


app = Flask(__name__)

# Import necessary libraries
import spacy
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import pandas as pd
# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def extract_entities(text):
    """Extract named entities from legal text"""
    doc = nlp(text)
    entities = {}
    
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        if ent.text not in entities[ent.label_]:
            entities[ent.label_].append(ent.text)
    
    return entities


def extract_key_sections(text):
    """Identify key sections in a legal document"""
    sections = {}
    
    # Common section patterns in legal documents
    patterns = {
        "definitions": r"(?i)(?:definitions|defined terms).*?(?=\n\n)",
        "obligations": r"(?i)(?:obligations|responsibilities|shall).*?(?=\n\n)",
        "termination": r"(?i)(?:termination|term|expiration).*?(?=\n\n)",
        "governing_law": r"(?i)(?:governing law|jurisdiction).*?(?=\n\n)",
        "warranties": r"(?i)(?:warranties|representations).*?(?=\n\n)"
    }
    
    for section_name, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            sections[section_name] = matches
    
    return sections

def generate_summary(text, ratio=0.2):
    """Generate a summary of the legal document"""
    sentences = sent_tokenize(text)
    
    # Create a TF-IDF vectorizer to find important sentences
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Calculate the importance of each sentence
    sentence_scores = tfidf_matrix.sum(axis=1).tolist()
    
    # Get the indices of the most important sentences
    num_sentences = max(1, int(len(sentences) * ratio))
    top_indices = sorted(range(len(sentence_scores)), 
                          key=lambda i: sentence_scores[i], 
                          reverse=True)[:num_sentences]
    
    # Sort indices to maintain original order
    top_indices.sort()
    
    # Construct summary from top sentences
    summary = " ".join([sentences[i] for i in top_indices])
    
    return summary


@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    legal_text = ""
    if request.method == "POST":
        legal_text = request.form.get("legal_text", "")
        cleaned_text = normalize_text(legal_text)

        entities = extract_entities(cleaned_text)
        key_sections = extract_key_sections(cleaned_text)
        summary = generate_summary(cleaned_text)

        results = {
            "entities": entities,
            "key_sections": key_sections,
            "summary": summary,
        }

    return render_template("index.html", results=results, legal_text=legal_text)


if __name__ == "__main__":
    # Running in debug mode for rapid iteration; disable in production.
    app.run(debug=True)
