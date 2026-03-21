from src.model_setup import load_model
import torch

def main():
    tokenizer, model = load_model()
    print("Tokenizer ja mudel laetud edukalt.\n")

    texts = [
        "See on näidistekst digiteeritud Eesti ajalehest.",
        "Gorbatšov päri Eesti plaanidega rahvarinde kongressil."
    ]

    for i, text in enumerate(texts):
        print(f"--- Näide {i+1} ---")
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)

        print("Tekst:             ", text)
        print("input_ids kuju:    ", inputs["input_ids"].shape)
        print("attention_mask kuju:", inputs["attention_mask"].shape)
        print("logits kuju:       ", outputs.logits.shape)
        print("logits väärtused:  ", outputs.logits)
        print()

if __name__ == "__main__":
    main()