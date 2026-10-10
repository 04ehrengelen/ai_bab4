
# Modified Example 4.22 - NLTK.py
# Text Analysis using Natural Language Processing

import nltk

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

# New text for analysis
sentence = """
Computer vision is a field of artificial intelligence
that enables computers to understand images and videos.
OpenCV is a popular library used by developers to process
images and detect objects.
Researchers at Stanford University develop advanced
machine learning techniques for image recognition.
"""

# 1. Tokenization
tokens = nltk.word_tokenize(sentence)
print("TOKENS:")
print(tokens)

# 2. Part-of-Speech Tagging
tagged = nltk.pos_tag(tokens)
print("\nPART-OF-SPEECH TAGGING:")
print(tagged)

# 3. Named Entity Recognition
entities = nltk.chunk.ne_chunk(tagged)
print("\nNAMED ENTITY RECOGNITION:")
print(entities)
