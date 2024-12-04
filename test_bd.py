import pymongo

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
