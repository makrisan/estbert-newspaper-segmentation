from src.model_setup import load_model
import torch

print("1. Skript käivitus")

sample_text = "RIIGIKOGUS ARUTATI EILE UUT SEADUST"
print("2. Näidistekst olemas")

tokenizer, model = load_model()
print("3. Tokenizer ja mudel laetud")

inputs = tokenizer(
    sample_text,
    return_tensors="pt",
    truncation=True,
    padding=True
)
print("4. Tekst tokeniseeritud")

outputs = model(**inputs)
print("5. Forward pass tehtud")

print("Input IDs:", inputs["input_ids"])
print("Attention mask:", inputs["attention_mask"])
print("Logits:", outputs.logits)
predicted_label = torch.argmax(outputs.logits, dim=1)
print("Predicted label:", predicted_label.item())