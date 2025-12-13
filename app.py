from collections import Counter
import re

from flask import Flask, render_template, request

app = Flask(__name__)

# Basic keywords often seen as section headers in contracts.
SECTION_KEYWORDS = [
    "definitions",
    "scope",
    "term",
    "termination",
    "confidentiality",
    "governing law",
    "payment",
    "liability",
    "indemnification",
    "representations",
    "warranties",
    "notices",
    "dispute resolution",
]

# Quick stopword list to help the summarizer focus on informative words.
STOPWORDS = {
    "the",
    "and",
    "of",
    "to",
    "a",
    "in",
    "for",
    "with",
    "on",
    "by",
    "an",
    "be",
    "is",
    "as",
    "that",
    "this",
    "at",
    "or",
    "from",
    "are",
    "was",
    "were",
    "it",
    "its",
    "which",
    "has",
    "have",
    "had",
    "not",
    "may",
    "can",
    "shall",
    "will",
    "such",
}


def normalize_text(legal_text: str) -> str:
    """Normalize text for downstream heuristics."""
    return legal_text.strip()


def extract_entities(legal_text: str) -> dict:
    """Lightweight entity extraction using regex heuristics."""
    dates = re.findall(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
        r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|"
        r"Nov(?:ember)?|Dec(?:ember)?)[\s.-]+\d{1,2},?\s+\d{4}\b",
        legal_text,
        flags=re.IGNORECASE,
    )
    amounts = re.findall(r"\$\s?\d[\d,]*(?:\.\d{2})?", legal_text)
    sections = re.findall(r"\bSection\s+\d+(?:\.\d+)*", legal_text, flags=re.IGNORECASE)

    # Capitalized word sequences as a simple proxy for party names or titles.
    candidate_parties = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", legal_text)
    parties = list({party for party in candidate_parties if len(party.split()) <= 5})

    return {
        "dates": sorted(set(dates)),
        "amounts": sorted(set(amounts)),
        "sections": sorted(set(sections)),
        "parties": sorted(parties),
    }


def extract_key_sections(legal_text: str) -> list:
    """Surface likely key clauses by looking for known section keywords."""
    key_sections = []
    for keyword in SECTION_KEYWORDS:
        if keyword in legal_text.lower():
            # Grab a short snippet around the keyword for context.
            match = re.search(
                r".{0,60}" + re.escape(keyword) + r".{0,120}",
                legal_text,
                flags=re.IGNORECASE,
            )
            snippet = match.group(0) if match else keyword
            key_sections.append(snippet.strip())
    return key_sections


def generate_summary(legal_text: str, max_sentences: int = 3) -> str:
    """Simple frequency-based summarizer to keep things dependency-free."""
    sentences = re.split(r"(?<=[.!?])\s+", legal_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return ""
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    words = re.findall(r"\b[a-zA-Z]+\b", legal_text.lower())
    freq = Counter(w for w in words if w not in STOPWORDS)
    if not freq:
        return " ".join(sentences[:max_sentences])

    scores = []
    for idx, sentence in enumerate(sentences):
        sentence_words = re.findall(r"\b[a-zA-Z]+\b", sentence.lower())
        score = sum(freq.get(w, 0) for w in sentence_words)
        scores.append((idx, score))

    top_indices = sorted(scores, key=lambda x: x[1], reverse=True)[:max_sentences]
    selected = sorted(idx for idx, _ in top_indices)
    return " ".join(sentences[idx] for idx in selected)


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
