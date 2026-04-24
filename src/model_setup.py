from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "tartuNLP/EstBERT"

# AutoModel = annab ainult “tekstiesituse” (embeddingud)
# AutoModelForSequenceClassification = annab kohe klassi (0 või 1)
def load_model(model_path=None):
    model_source = model_path if model_path else MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        num_labels=2  # 0 või 1
    )
    return tokenizer, model
