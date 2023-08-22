import datetime


def get_date(date: str):
    return datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%f")


def get_date_audio(date: str):
    return datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")

