import torch
from torch import nn


class SalaryPredictionModel(nn.Module):
    def __init__(self, bert, num_employers, num_exp_levels, num_cities, employer_emb_dim=16, exp_emb_dim=4,
                 city_emb_dim=16, hidden_dim=128):
        super(SalaryPredictionModel, self).__init__()
        self.bert = bert
        hidden_size = self.bert.config.hidden_size
        self.employer_emb = nn.Embedding(num_employers, employer_emb_dim)
        self.exp_emb = nn.Embedding(num_exp_levels, exp_emb_dim)
        self.city_emb = nn.Embedding(num_cities, city_emb_dim)

        # Теперь в cat пойдут CLS + employer_emb + exp_emb + city_emb
        total_input_dim = hidden_size + employer_emb_dim + exp_emb_dim + city_emb_dim
        self.fc1 = nn.Linear(total_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, employer, experience, city):
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Используем CLS-токен
        cls_token = bert_outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]

        employer_emb = self.employer_emb(employer)
        exp_emb = self.exp_emb(experience)
        city_emb = self.city_emb(city)

        x = torch.cat([cls_token, employer_emb, exp_emb, city_emb], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.output(x).squeeze()
