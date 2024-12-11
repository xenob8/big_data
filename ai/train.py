import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from ai.data import descriptions, salaries, experience_levels, cities, employers, train_employers, test_employers, \
    employer_to_idx, train_exps, exp_to_idx, test_exps, city_to_idx, train_cities, test_cities, train_desc, test_desc, \
    train_salaries, test_salaries, unique_employers, unique_exp_levels, unique_cities
from ai.model import SalaryPredictionModel
from test_bd import get_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_employer_indices = np.array(
    [employer_to_idx[e] if e in employer_to_idx else employer_to_idx["unknown"] for e in train_employers],
    dtype=np.int32
)
test_employer_indices = np.array(
    [employer_to_idx[e] if e in employer_to_idx else employer_to_idx["unknown"] for e in test_employers],
    dtype=np.int32
)

train_exp_indices = np.array([exp_to_idx[e] for e in train_exps], dtype=np.int32)
test_exp_indices = np.array([exp_to_idx[e] for e in test_exps], dtype=np.int32)

train_city_indices = np.array(
    [city_to_idx[c] if c in city_to_idx else city_to_idx["unknown_city"] for c in train_cities],
    dtype=np.int32
)
test_city_indices = np.array(
    [city_to_idx[c] if c in city_to_idx else city_to_idx["unknown_city"] for c in test_cities],
    dtype=np.int32
)

model_name = "cointegrated/rubert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)

max_len = 128
train_tokens = tokenizer(train_desc, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
test_tokens = tokenizer(test_desc, truncation=True, padding=True, max_length=max_len, return_tensors="pt")


class SalaryDataset(Dataset):
    def __init__(self, tokens, salaries, employer_indices, exp_indices, city_indices):
        self.tokens = tokens
        self.salaries = torch.tensor(salaries, dtype=torch.float32)
        self.employer_indices = torch.tensor(employer_indices, dtype=torch.long)
        self.exp_indices = torch.tensor(exp_indices, dtype=torch.long)
        self.city_indices = torch.tensor(city_indices, dtype=torch.long)

    def __len__(self):
        return len(self.salaries)

    def __getitem__(self, idx):
        return {
            "input_ids": self.tokens["input_ids"][idx],
            "attention_mask": self.tokens["attention_mask"][idx],
            "employer": self.employer_indices[idx],
            "experience": self.exp_indices[idx],
            "city": self.city_indices[idx],
            "salary": self.salaries[idx]
        }


train_dataset = SalaryDataset(train_tokens, train_salaries, train_employer_indices, train_exp_indices,
                              train_city_indices)
test_dataset = SalaryDataset(test_tokens, test_salaries, test_employer_indices, test_exp_indices, test_city_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Размораживаем последние два слоя BERT
for param in bert_model.parameters():
    param.requires_grad = False

encoder_layers = bert_model.encoder.layer
for layer in encoder_layers[-2:]:
    for param in layer.parameters():
        param.requires_grad = True

model = SalaryPredictionModel(bert_model, len(unique_employers), len(unique_exp_levels), len(unique_cities))
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

num_epochs = 50
num_training_steps = num_epochs * len(train_loader)
num_warmup_steps = int(0.1 * num_training_steps)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=num_warmup_steps,
                                            num_training_steps=num_training_steps)

criterion = nn.MSELoss()


def train_epoch(model, dataloader, optimizer, criterion, scheduler):
    model.train()
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        employer = batch["employer"].to(device)
        experience = batch["experience"].to(device)
        city = batch["city"].to(device)
        salaries = batch["salary"].to(device)
        outputs = model(input_ids, attention_mask, employer, experience, city)
        loss = criterion(outputs, salaries)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            employer = batch["employer"].to(device)
            experience = batch["experience"].to(device)
            city = batch["city"].to(device)
            salaries = batch["salary"].to(device)
            outputs = model(input_ids, attention_mask, employer, experience, city)
            loss = criterion(outputs, salaries)
            total_loss += loss.item()
    return total_loss / len(dataloader)


for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler)
    val_loss = evaluate(model, test_loader, criterion)
    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

os.makedirs("saved_model", exist_ok=True)
torch.save(model.state_dict(), "saved_model/salary_prediction_model.pt")
print("Model saved successfully.")
