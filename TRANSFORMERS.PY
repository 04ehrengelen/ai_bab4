
# Example 4.23 - Sentiment Analysis
# Using Hugging Face Transformers

from transformers import pipeline

# Membuat pipeline untuk analisis sentimen
classifier = pipeline("sentiment-analysis")

# Teks yang akan dianalisis
result = classifier("This is a good movie.")

# Menampilkan hasil
print(result)
print("Sentiment:", result[0]["label"])
print("Confidence:", round(result[0]["score"] * 100, 2), "%")
