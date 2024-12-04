from constants import deviation_percent


def calc_upper_bound(lower_bound):
    return lower_bound * (1 + deviation_percent)


def calc_lower_bound(upper_bound):
    return upper_bound / (1 + deviation_percent)


def get_avg_salary(data):
    currency = data["salary"]["currency"]
    upper_bound_salary = data["salary"]["to"]
    lower_bound_salary = data["salary"]["from"]

    if currency == "USD":
        upper_bound_salary = upper_bound_salary * 100
        lower_bound_salary = lower_bound_salary * 100
    elif currency == "EUR":
        upper_bound_salary = upper_bound_salary * 100
        lower_bound_salary = lower_bound_salary * 105

    if not upper_bound_salary:
        upper_bound_salary = calc_lower_bound(lower_bound_salary)
    elif not lower_bound_salary:
        lower_bound_salary = calc_lower_bound(upper_bound_salary)

    return (upper_bound_salary + lower_bound_salary) / 2
