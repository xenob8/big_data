import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, kstest
from bd import vac_col
from utils import get_avg_salary

datas = vac_col.find({
    "$and": [
        {"salary": {"$ne": None}},
    ]
})
items = []
salaries = []
for data in datas:
    item = {}
    salary = None
    if data["salary"]:
        salary = get_avg_salary(data)
        if salary < 10_000:
            continue
    item["salary"] = salary
    city = data["area"]["name"]
    item["city"] = city

    description = f"{(data["snippet"]["requirement"] or "")}  {(data["snippet"]["responsibility"] or "")} + {data["name"]}"

    item["descr"] = description
    item["exp"] = data["experience"]
    item["employer"] = data["employer"]["name"]
    items.append(item)

    salaries.append(salary)
# print(salaries)
salaries = np.array(salaries)
salaries = np.log(salaries)
# sns.histplot(salaries, kde=True)
stat, p = shapiro(salaries)
print(f"Статистика теста: {stat}, p-значение: {p}")
if p > 0.05:
    print("Распределение похоже на нормальное (не отвергаем H0)")
else:
    print("Распределение не является нормальным (отвергаем H0)")

stat, p = normaltest(salaries)
print(f"Статистика теста: {stat}, p-значение: {p}")
if p > 0.05:
    print("Распределение похоже на нормальное (не отвергаем H0)")
else:
    print("Распределение не является нормальным (отвергаем H0)")

stat, p = kstest(salaries, 'norm', args=(salaries.mean(), salaries.std()))
print(f"Статистика теста: {stat}, p-значение: {p}")
if p > 0.05:
    print("Распределение похоже на нормальное (не отвергаем H0)")
else:
    print("Распределение не является нормальным (отвергаем H0)")

stats.probplot(salaries, dist="norm", plot=plt)
# plt.title("Q-Q график")
# plt.xlabel("Логарифмическая зарплата")
# plt.ylabel("Частота")
# plt.title("Гистограмма логарифмеческих зарплат с графиком плотности")

plt.xlabel("Теоретические квартили", fontsize=12)  # Подпись по оси X
plt.ylabel("Наблюдаемые значения", fontsize=12)    # Подпись по оси Y
plt.title("Вероятностный график", fontsize=14)               # Заголовок
plt.grid(True)
plt.legend(["Точки данных", "Теоретическая линия"], loc="upper left", fontsize=10)
# plt.legend(labels=["Оцененная плотность данных", "Гистограмма"], loc="upper right")
plt.show()
# plt.show()