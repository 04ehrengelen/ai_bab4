
# Example 4.23 - Modified Sentiment Analysis
# Using Hugging Face Transformers

from transformers import pipeline

# Membuat pipeline analisis sentimen
classifier = pipeline("sentiment-analysis")

# Kalimat untuk dianalisis
sentences = [
    "This is a wonderful movie.",
    "I really enjoyed this film.",
    "The acting was excellent and the story was amazing.",
    "This movie is terrible.",
    "I hate this film because it is boring.",
    "The story was disappointing and the acting was poor."
]

# Melakukan analisis sentimen
for sentence in sentences:
    result = classifier(sentence)[0]

    print("Sentence:", sentence)
    print("Sentiment:", result["label"])
    print("Confidence:", round(result["score"] * 100, 2), "%")
    print("-" * 60)
