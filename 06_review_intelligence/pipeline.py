"""
Project 06: Review Intelligence Engine
NLP pipeline: sentiment analysis + theme extraction + LLM recommendations
from large-scale customer review datasets.
"""

import re
import json
from collections import Counter, defaultdict
from typing import Optional
from dataclasses import dataclass, field
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import torch

DEVICE = 0 if torch.cuda.is_available() else -1


@dataclass
class ReviewAnalysis:
    total_reviews: int
    avg_sentiment_score: float
    sentiment_distribution: dict
    top_themes: list[dict]
    theme_sentiment: dict
    recommendations: list[str]
    critical_issues: list[str]


class ReviewIntelligenceEngine:
    def __init__(self):
        print("⏳ Loading sentiment model...")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=DEVICE,
            truncation=True,
            max_length=512,
        )

        print("⏳ Loading sentence encoder for theme extraction...")
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

        print("⏳ Loading text generation for recommendations...")
        self.llm = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=DEVICE,
            max_new_tokens=200,
        )

        # Theme keywords for customer reviews
        self.theme_keywords = {
            "delivery": ["shipping", "delivery", "arrived", "package", "late", "fast", "slow"],
            "quality": ["quality", "material", "durable", "cheap", "excellent", "poor", "broken"],
            "customer_service": ["support", "service", "agent", "helpful", "rude", "responsive"],
            "pricing": ["price", "expensive", "cheap", "value", "cost", "worth", "overpriced"],
            "usability": ["easy", "simple", "confusing", "interface", "setup", "difficult", "intuitive"],
            "features": ["feature", "function", "capability", "option", "missing", "lacks"],
            "packaging": ["packaging", "box", "wrapped", "damaged", "protected"],
            "performance": ["fast", "slow", "performance", "speed", "lag", "efficient"],
        }
        print("✅ Review Intelligence Engine ready.")

    def analyze_sentiment(self, texts: list[str]) -> list[dict]:
        """Batch sentiment analysis using RoBERTa."""
        results = self.sentiment_pipeline(texts, batch_size=32)
        normalized = []
        for r in results:
            label = r["label"].lower()
            score = r["score"]
            # Normalize to (-1, 0, 1) scale
            if "positive" in label:
                normalized.append({"sentiment": "positive", "score": score, "numeric": score})
            elif "negative" in label:
                normalized.append({"sentiment": "negative", "score": score, "numeric": -score})
            else:
                normalized.append({"sentiment": "neutral", "score": score, "numeric": 0.0})
        return normalized

    def extract_themes(self, texts: list[str]) -> dict[str, list[int]]:
        """Rule-based theme extraction: returns {theme: [review_indices]}."""
        theme_hits: dict[str, list[int]] = defaultdict(list)
        for i, text in enumerate(texts):
            lower = text.lower()
            for theme, keywords in self.theme_keywords.items():
                if any(kw in lower for kw in keywords):
                    theme_hits[theme].append(i)
        return dict(theme_hits)

    def extract_key_phrases(self, texts: list[str], top_n: int = 20) -> list[tuple[str, int]]:
        """Extract frequent noun phrases / bigrams as recurring topics."""
        # Simple bigram extraction
        all_bigrams = []
        for text in texts:
            words = re.findall(r"\b[a-z]{3,}\b", text.lower())
            bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
            all_bigrams.extend(bigrams)
        # Filter stop bigrams
        stop_words = {"the the", "and the", "is the", "in the", "to the", "of the", "it is", "i am", "was the"}
        counts = Counter(b for b in all_bigrams if b not in stop_words and len(b) > 5)
        return counts.most_common(top_n)

    def generate_recommendations(self, theme_sentiments: dict[str, float]) -> list[str]:
        """Use Flan-T5 to generate actionable recommendations from theme data."""
        weak_themes = [(t, s) for t, s in theme_sentiments.items() if s < 0]
        if not weak_themes:
            return ["Maintain current quality across all areas — customer sentiment is positive."]

        recommendations = []
        for theme, score in sorted(weak_themes, key=lambda x: x[1])[:5]:
            prompt = (
                f"A product has negative customer reviews about {theme.replace('_', ' ')}. "
                f"Suggest one specific, actionable improvement for the product team."
            )
            result = self.llm(prompt)
            recommendations.append(f"[{theme.upper()}] {result[0]['generated_text'].strip()}")
        return recommendations

    def find_critical_issues(self, texts: list[str], sentiments: list[dict]) -> list[str]:
        """Surface the most negative review excerpts as critical issues."""
        critical = []
        for text, sent in zip(texts, sentiments):
            if sent["numeric"] < -0.7:
                # Extract most negative sentence
                sentences = re.split(r"[.!?]", text)
                if sentences:
                    critical.append(sentences[0].strip()[:200])
        return list(set(critical))[:10]

    def analyze(self, reviews: list[str]) -> ReviewAnalysis:
        print(f"🔍 Analyzing {len(reviews)} reviews...")

        sentiments = self.analyze_sentiment(reviews)
        avg_score = np.mean([s["numeric"] for s in sentiments])
        dist = Counter(s["sentiment"] for s in sentiments)

        themes = self.extract_themes(reviews)
        top_themes = sorted(
            [{"theme": t, "count": len(idxs)} for t, idxs in themes.items()],
            key=lambda x: -x["count"],
        )

        # Compute average sentiment per theme
        theme_sentiment = {}
        for theme, idxs in themes.items():
            if idxs:
                theme_sentiment[theme] = float(np.mean([sentiments[i]["numeric"] for i in idxs]))

        recommendations = self.generate_recommendations(theme_sentiment)
        critical = self.find_critical_issues(reviews, sentiments)

        return ReviewAnalysis(
            total_reviews=len(reviews),
            avg_sentiment_score=round(float(avg_score), 3),
            sentiment_distribution=dict(dist),
            top_themes=top_themes[:10],
            theme_sentiment=theme_sentiment,
            recommendations=recommendations,
            critical_issues=critical,
        )

    def analyze_dataframe(self, df: pd.DataFrame, text_col: str) -> ReviewAnalysis:
        """Analyze from a pandas DataFrame."""
        return self.analyze(df[text_col].dropna().tolist())


# ─── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = ReviewIntelligenceEngine()
    sample_reviews = [
        "Great quality product! Delivery was fast and packaging was perfect.",
        "Terrible customer service. Waited 3 weeks and the package was damaged.",
        "The price is reasonable for what you get. Setup was a bit confusing.",
        "Excellent performance and the features are exactly what I needed.",
        "Very slow shipping and the product looks cheap. Not worth the price.",
        "Customer support was super helpful when I had an issue. Resolved quickly.",
        "Poor quality materials. Broke after a week. Complete waste of money.",
        "Easy to use, intuitive interface. Highly recommend!",
    ]
    result = engine.analyze(sample_reviews)
    print(f"\n📊 Avg Sentiment: {result.avg_sentiment_score}")
    print(f"📈 Distribution: {result.sentiment_distribution}")
    print(f"\n🏷️ Top Themes: {[t['theme'] for t in result.top_themes[:5]]}")
    print(f"\n💡 Recommendations:")
    for r in result.recommendations:
        print(f"  • {r}")
