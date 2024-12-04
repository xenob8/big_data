import pymongo

myclient = pymongo.MongoClient("mongodb://localhost:27017/")

mydb = myclient["mydatabase"]
vac_col = mydb["vac"]


def deviation_percent(upper_bounds, lower_bounds):
    upper_bound_sum = sum(upper_bounds)
    lower_bound_sum = sum(lower_bounds)
    delta = (upper_bound_sum - lower_bound_sum) / lower_bound_sum * 100
    return delta


def calc_deviation_percent():
    datas = vac_col.find({
        "$and": [
            {"salary": {"$ne": None}},
            {"salary.from": {"$ne": None}},
            {"salary.to": {"$ne": None}}
        ]
    })
    upper_bounds = []
    lower_bounds = []
    for data in datas:
        currency = data["salary"]["currency"]
        upper_bound_salary = data["salary"]["to"]
        lower_bound_salary = data["salary"]["from"]
        if currency == "USD":
            upper_bound_salary = upper_bound_salary * 100
            lower_bound_salary = lower_bound_salary * 100
        elif currency == "EUR":
            upper_bound_salary = upper_bound_salary * 100
            lower_bound_salary = lower_bound_salary * 105
        # print(data)
        upper_bounds.append(upper_bound_salary)
        lower_bounds.append(lower_bound_salary)

    print(deviation_percent(upper_bounds, lower_bounds))
