import json

class ConfigReader():
    def __init__(self):
        with open("config/config.json") as config_dict:
            self.__dict__ = json.load(config_dict)
