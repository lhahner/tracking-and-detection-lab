import json
from pathlib import Path
from typing import TypeVar

T = TypeVar('T')


class Deserializer:
    def __init__(self, data_format="json"):
        self.data_format = data_format

    def deserialize(self, into_or_serialized, serialized=None):
        if self.data_format != "json":
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")

        if serialized is None:
            return self.__deserialize_document(into_or_serialized)

        into = into_or_serialized
        raw = json.loads(serialized)

        from entities.detection import Detection

        if into is Detection:
            return self.__decode_detection_value(value=raw)

        raise ValueError(f"Cannot deserialize into {into}")

    def __deserialize_document(self, source):
        source_path = Path(source)
        if source_path.exists():
            return json.loads(source_path.read_text(encoding="utf-8"))
        breakpoint()
        return json.loads(source)

    def __decode_detection_value(self, value):
        import torch
        from entities.detection import Detection

        return Detection(score=torch.tensor(value["score"]),
                         label=value["label"],
                         box=torch.tensor(value["box"]))
