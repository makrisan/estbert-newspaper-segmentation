from src.model_setup import load_model
from src.dataset import load_jsonl
import torch

DATA_PATH = "data/sample/sample_dataset.jsonl"

print("1. Skript käivitus")

# loe andmed failist sisse
data = load_jsonl(DATA_PATH)
print("2. Andmed loetud failist sisse")
print("Näidete arv:", len(data))

# võta esimene näide datasetist
sample_text = data[0]["text"]
sample_label = data[0]["label"]

print("3. Esimene näidistekst:", sample_text)
print("4. Esimese näite label:", sample_label)

# lae tokenizer ja mudel
tokenizer, model = load_model()
print("5. Tokenizer ja mudel laetud")

# tokeniseeri tekst
inputs = tokenizer(
    sample_text,
    return_tensors="pt",
    truncation=True,
    padding=True
)
print("6. Tekst tokeniseeritud")

# tee forward pass
outputs = model(**inputs)
print("7. Forward pass tehtud")

# prindi väljund
print("Input IDs:", inputs["input_ids"])
print("Attention mask:", inputs["attention_mask"])
print("Logits:", outputs.logits)

predicted_label = torch.argmax(outputs.logits, dim=1)
print("Predicted label:", predicted_label.item())
