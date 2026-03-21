from src.dataset import load_jsonl, NewsDataset
from src.model_setup import load_model
import torch, os
from torch.utils.data import DataLoader

# --- PATHID ---
TRAIN_PATH = "data/sample/train.jsonl"
VAL_PATH = "data/sample/val.jsonl"

# --- HYPERPARAMETERS ---
BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 2e-5

def main():
    print("1. Alustan treeningut")

    # --- LOAD DATA ---
    train_data = load_jsonl(TRAIN_PATH)
    val_data = load_jsonl(VAL_PATH)

    print("Train size:", len(train_data))
    print("Val size:", len(val_data))

    # --- LOAD MODEL ---
    tokenizer, model = load_model()

    # --- DATASET ---
    train_dataset = NewsDataset(train_data, tokenizer)
    val_dataset = NewsDataset(val_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # --- DEVICE ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- OPTIMIZER ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # --- TRAIN LOOP ---
    model.train()

    for epoch in range(EPOCHS):
        print(f"\nEPOCH {epoch+1}")

        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print("Train loss:", avg_loss)

    # --- SAVE MODEL ---
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model.pt")
    print("Mudeli checkpoint salvestatud!")

if __name__ == "__main__":
    main()
