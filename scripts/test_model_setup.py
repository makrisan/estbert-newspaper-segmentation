from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "tartuNLP/EstBERT"

def main():
    print(f"Laen mudelit: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  ## Laeb tokenizeri
    model = AutoModel.from_pretrained(MODEL_NAME) # Laeb mudeli

    text = "See on näidistekst digiteeritud Eesti ajalehest."  # Testandmed
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)  # Tokeniseerimine

    with torch.no_grad():  # Testime mudelit
        outputs = model(**inputs)

    print("Mudel laeti edukalt.")
    print("input_ids kuju:", inputs["input_ids"].shape)
    print("last_hidden_state kuju:", outputs.last_hidden_state.shape)

if __name__ == "__main__":
    main()