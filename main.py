from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

text = "I love learning Artificial Intelligence!"

result = classifier(text)

print("Input:", text)
print("Result:", result)