import numpy as np
from sklearn.model_selection import train_test_split

from test_bd import get_data

vacancies = get_data()
print("Total vacancies:", len(vacancies))

descriptions = [v["descr"] for v in vacancies]
salaries = np.array([v["salary"] for v in vacancies], dtype=np.float32)
employers = [v["employer"] for v in vacancies]
experience_levels = [v["exp"] for v in vacancies]
cities = [v["city"] for v in vacancies]


(
    train_desc,
    test_desc,
    train_salaries,
    test_salaries,
    train_employers,
    test_employers,
    train_exps,
    test_exps,
    train_cities,
    test_cities
) = train_test_split(
    descriptions, salaries, employers, experience_levels, cities,
    test_size=0.2, random_state=42
)

salary_mean = np.mean(train_salaries)
salary_std = np.std(train_salaries)
train_salaries = (train_salaries - salary_mean) / salary_std
test_salaries = (test_salaries - salary_mean) / salary_std

unique_employers = list(set(train_employers))
unique_employers.append("unknown")
employer_to_idx = {e: i for i, e in enumerate(unique_employers)}

unique_exp_levels = ['between3And6', 'moreThan6', 'between1And3', 'noExperience']
exp_to_idx = {e: i for i, e in enumerate(unique_exp_levels)}

unique_cities = list(set(train_cities))
unique_cities.append("unknown_city")
city_to_idx = {c: i for i, c in enumerate(unique_cities)}