import datetime
import json
from datetime import timedelta

import requests

from bd import vac_col
from my_secretes import TOKEN

url = "https://api.hh.ru/vacancies"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",  # Добавьте, если требуется
}

TOTAL_DAYS = 30
date_now = datetime.datetime.now()
date_from = date_now - datetime.timedelta(days=TOTAL_DAYS)
date_to = date_from

params_dict = {
    "professional_role": [96, 124],
    "page": 0,
    "date_from": date_from,
    "date_to": date_to,
    "per_page": 100,
}


def get_total_pages(data):
    return data.get("pages")


def save_data(data, date, page):
    # json_data = json.loads(data)
    with open(f"data/{date}_{page}.json", "w", encoding="UTF-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def save_in_mongo(data):
    items = data.get("items")

    vac_col.insert_many(items)


while date_to < date_now:
    date_from = date_to
    date_to = date_from + timedelta(days=1)
    print(f"iter {date_from} : {date_to}")
    params_dict["date_from"] = date_from.isoformat()
    params_dict["date_to"] = date_to.isoformat()
    print(date_to)
    data = requests.get(url=url, headers=headers, params=params_dict)
    data_json = data.json()
    total_pages = get_total_pages(data_json)
    # save_in_mongo(data.json())

    for page in range(0, total_pages):
        params_dict["page"] = page

        data = requests.get(url=url, headers=headers, params=params_dict)
        save_in_mongo(data.json())
        # save_data(data.json(), date_to, page)
        # print(data.json())

# print(data.json())
