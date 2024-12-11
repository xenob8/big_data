import datetime

import torch
from transformers import AutoTokenizer, AutoModel

from ai.data import unique_employers, unique_exp_levels, unique_cities, salary_std, salary_mean
from ai.model import SalaryPredictionModel

# Импортируем модель из обучающего модуля

# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Параметры
model_name = "cointegrated/rubert-tiny"
max_len = 128
saved_model_path = "saved_model/salary_prediction_model.pt"

# Загрузка токенизатора и модели BERT
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name)

# Загружаем параметры окружения из обучающего скрипта


employer_to_idx = {e: i for i, e in enumerate(unique_employers)}
exp_to_idx = {e: i for i, e in enumerate(unique_exp_levels)}
city_to_idx = {c: i for i, c in enumerate(unique_cities)}

# Инициализация модели
model = SalaryPredictionModel(
    bert_model,
    num_employers=len(unique_employers),
    num_exp_levels=len(unique_exp_levels),
    num_cities=len(unique_cities)
)
model.load_state_dict(torch.load(saved_model_path, map_location=device))
model.to(device)
model.eval()


def preprocess_input(description, employer, experience, city):
    """Функция для подготовки входных данных."""
    tokens = tokenizer(description, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    employer_idx = torch.tensor([employer_to_idx.get(employer, employer_to_idx["unknown"])], dtype=torch.long)
    exp_idx = torch.tensor([exp_to_idx.get(experience, exp_to_idx["noExperience"])], dtype=torch.long)
    city_idx = torch.tensor([city_to_idx.get(city, city_to_idx["unknown_city"])], dtype=torch.long)
    return tokens, employer_idx, exp_idx, city_idx


def predict_salary(description, employer, experience, city):
    """Инференс: предсказание зарплаты."""
    tokens, employer_idx, exp_idx, city_idx = preprocess_input(description, employer, experience, city)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    employer_idx = employer_idx.to(device)
    exp_idx = exp_idx.to(device)
    city_idx = city_idx.to(device)

    with torch.no_grad():
        output = model(input_ids, attention_mask, employer_idx, exp_idx, city_idx).item()
        actual_salary = output * salary_std + salary_mean
        return actual_salary


# Тестовый пример
if __name__ == "__main__":
    description = "Разработка и поддержка веб-приложений на Python."
    employer = "СБЕР"
    experience = "noExperience"
    city = "Москва"

    predicted_salary = predict_salary(description, employer, experience, city)
    last = datetime.datetime.now()
    predicted_salary = predict_salary(description, employer, experience, city)
    new = datetime.datetime.now()
    print(f"Predicted salary (normalized): {predicted_salary:.4f}")
    print(f"time {(new - last).microseconds}")
