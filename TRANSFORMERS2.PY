
from transformers import pipeline

# Load a pretrained Question Answering model using PyTorch
question_answerer = pipeline(
    "question-answering",
    model="distilbert/distilbert-base-cased-distilled-squad",
    framework="pt"
)

# Define the question and context
question = "What is the name of the company?"
context = "We created Biox Systems Ltd company back in the year of 2000."

# Find the answer from the given context
result = question_answerer(
    question=question,
    context=context
)

# Display the result
print("Question:", question)
print("Context:", context)
print("Answer:", result["answer"])
print("Confidence:", round(result["score"] * 100, 2), "%")
