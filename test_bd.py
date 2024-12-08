from elasticsearch import Elasticsearch, helpers

from bd import vac_col
from utils import get_avg_salary


def test_total_size():
    datas = vac_col.count_documents({}
        # "$and": [
        #     {"salary": {"$ne": None}},
        #     {"salary.from": {"$ne": None}},
        #     {"salary.to": {"$ne": None}}
        # ]}
    )
    print(datas)


def test_preload_elastic_search():
    es = Elasticsearch("http://localhost:9200")

    datas = vac_col.find({
        "$and": [
            {"salary": {"$ne": None}},
        ]
    })
    items = []
    for data in datas:
        item = {}
        salary = None
        if data["salary"]:
            salary = get_avg_salary(data)
        item["salary"] = salary
        city = data["area"]["name"]
        item["city"] = city

        description = f"{(data["snippet"]["requirement"] or "")}  {(data["snippet"]["responsibility"] or "")} + {data["name"]}"

        item["descr"] = description
        item["exp"] = data["experience"]
        item["employer"] = data["employer"]["name"]
        items.append(item)

    actions = [
        {"_index": "with_salary",
         "_source": doc,
         "pipeline": "lang_detector"
         } for doc in items
    ]
    helpers.bulk(es, actions)


def test_get_all_adresses():
    adresses = set()
    currencies = set()
    datas = vac_col.find({
        "$and": [
            {"salary": {"$ne": None}},
        ]
    })

    salaries = []
    for data in datas:
        salary = get_avg_salary(data)
        salaries.append(salary)

    print(sum(salaries) / len(salaries))


def test_elastic():
    es = Elasticsearch("http://localhost:9200")
    query = {

        "query": {
            "match": {
                "descr": "javascript"
            }
        }
    }
    resp = es.search(index="vacancies", body=query)
    print(resp)


def test_add_pipeline():
    # Подключение к Elasticsearch
    es = Elasticsearch("http://localhost:9200")  # Укажите ваш URL Elasticsearch

    # Определение пайплайна
    pipeline_body = {
        "processors": [
            {
                "script": {
                    "source": """
                        String desc = ctx.descr.toLowerCase();

                        if (desc.contains("java")) {
                            ctx.lang = "Java";
                        } else if (desc.contains("python")) {
                            ctx.lang = "Python";
                        } else if (desc.contains("1c") || desc.contains("1с")) {
                            ctx.lang = "1c";
                        } else if (desc.contains("javascript") || desc.contains("js") || desc.contains("nodejs") || desc.contains("typescript") || desc.contains("ts") ) {
                            ctx.lang = "JS|TS";
                        } else if (desc.contains("c++") || desc.contains("cpp")) {
                            ctx.lang = "C++";
                        } else if (desc.contains("c#") || desc.contains("c-sharp")) {
                            ctx.lang = "C#";
                        } else if (desc.contains("ruby")) {
                            ctx.lang = "Ruby";
                        } else if (desc.contains("php")) {
                            ctx.lang = "PHP";
                        } else if (desc.contains("go") || desc.contains("golang")) {
                            ctx.lang = "Go";
                        } else if (desc.contains("kotlin")) {
                            ctx.lang = "Kotlin";
                        } else if (desc.contains("swift")) {
                            ctx.lang = "Swift";
                        } else if (desc.contains("rust")) {
                            ctx.lang = "Rust";
                        } else if (desc.contains("scala")) {
                            ctx.lang = "Scala";
                        } else if (desc.contains("perl")) {
                            ctx.lang = "Perl";
                        } else {
                            ctx.lang = "Other";
                        }
                    """
                }
            }
        ]
    }

    # Создание пайплайна
    response = es.ingest.put_pipeline(id="lang_detector", body=pipeline_body)

    print(response)  # Вывод ответа от Elasticsearch


def test_delete_index():
    es = Elasticsearch("http://localhost:9200")  # Укажите ваш URL Elasticsearch
    es.delete_by_query(index="all", body={
        "query": {
            "match_all": {}  # Удалить все документы или добавить более узкий фильтр
        }
    })
