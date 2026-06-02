import json, torch
from typing import TypeVar
from entities.detection import Detection

T = TypeVar('T')


class Deserializer:
    def __init__(self, data_format="json"):
        self.data_format = data_format

    def deserialize(self, into: type[T], serialized):
        raw = json.loads(serialized)
        if into is Detection:
            return self.__decode_detection_value(value=raw)

    def __decode_detection_value(self, value) -> Detection:
        return Detection(score=torch.tensor(value["score"]),
                         box=torch.tensor(value["box"]))
