import pymongo
from elasticsearch import Elasticsearch, helpers

from bd import vac_col
from utils import get_avg_salary


def test_total_size():
    datas = vac_col.count_documents({
        "$and": [
            {"salary": {"$ne": None}},
            {"salary.from": {"$ne": None}},
            {"salary.to": {"$ne": None}}
        ]}
    )
    print(datas)


def preload_elastic_search():
    es = Elasticsearch("http://localhost:9200")

    datas = vac_col.find({
        "$and": [
            {"salary": {"$ne": None}},
        ]
    })
    items = []
    for data in datas:
        item = {}
        salary = get_avg_salary(data)
        item["salary"] = salary
        city = data["area"]["name"]
        item["city"] = city
        items.append(item)

    actions = [
        {"_index": "salaries", "_source": doc} for doc in items
    ]
    helpers.bulk(es, actions)
