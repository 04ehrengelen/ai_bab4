
from transformers import pipeline

# Load a pretrained Question Answering model using PyTorch
question_answerer = pipeline(
    "question-answering",
    model="distilbert/distilbert-base-cased-distilled-squad",
    framework="pt"
)

# Define a new context about artificial intelligence
context = (
    "Artificial intelligence is a technology that enables computers "
    "to perform tasks that normally require human intelligence. "
    "Computer vision is a field of AI that allows computers to analyze "
    "images and videos. OpenCV is a popular library used for computer "
    "vision applications. Researchers at Stanford University have "
    "contributed to the development of artificial intelligence."
)

# Prepare different questions about the context
questions = [
    "What does artificial intelligence enable computers to do?",
    "What is computer vision?",
    "Which library is popular for computer vision applications?",
    "Where have researchers contributed to AI development?"
]

# Answer each question based on the given context
for question in questions:
    result = question_answerer(
        question=question,
        context=context
    )

    # Display the question, answer, and confidence score
    print("Question:", question)
    print("Answer:", result["answer"])
    print("Confidence:", round(result["score"] * 100, 2), "%")
    print("-" * 60)
