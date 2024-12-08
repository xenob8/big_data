from constants import deviation_percent


def calc_upper_bound(lower_bound):
    return lower_bound * (1 + deviation_percent)


def calc_lower_bound(upper_bound):
    return upper_bound / (1 + deviation_percent)


def get_avg_salary(data):
    currency = data["salary"]["currency"]
    upper_bound_salary = data["salary"]["to"]
    lower_bound_salary = data["salary"]["from"]

    if not upper_bound_salary:
        upper_bound_salary = calc_lower_bound(lower_bound_salary)
    elif not lower_bound_salary:
        lower_bound_salary = calc_lower_bound(upper_bound_salary)

    middle_salary = (upper_bound_salary + lower_bound_salary) / 2

    return price_to_rub(middle_salary, currency)


def price_to_rub(price, currency):
    currency_rates = {
        'RUR': 1,
        'USD': 99.4215,
        'EUR': 106.3040,
        'AZN': 58.4832,
        'KZT': 0.1895,
        'KGS': 0.1145,
        'UZS': 0.0077,
        'BYN': 29.1858,
        "BYR": 29.1858 / 10_000
    }
    return currency_rates[currency] * price

