import pymongo

from utils import get_avg_salary

myclient = pymongo.MongoClient("mongodb://localhost:27017/")

mydb = myclient["mydatabase"]
vac_col = mydb["vac"]


def test_total_size():
    datas = vac_col.count_documents({
        "$and": [
            {"salary": {"$ne": None}},
            {"salary.from": {"$ne": None}},
            {"salary.to": {"$ne": None}}
        ]}
    )
    print(datas)


def test_avg_salary():
    datas = vac_col.find({
        "$and": [
            {"salary": {"$ne": None}},
        ]
    })
    salaries = []
    for data in datas:
        salary = get_avg_salary(data)
        salaries.append(salary)
    avg = sum(salaries) / len(salaries)
    print(avg)
