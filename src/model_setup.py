from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "tartuNLP/EstBERT"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    return tokenizer, model