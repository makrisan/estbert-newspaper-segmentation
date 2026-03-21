from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "tartuNLP/EstBERT"

# AutoModel = annab ainult “tekstiesituse” (embeddingud)
# AutoModelForSequenceClassification = annab kohe klassi (0 või 1)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2  # 0 või 1
    )
    return tokenizer, model
