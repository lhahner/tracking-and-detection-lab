import json
from pathlib import Path


class Deserializer:
    def __init__(self, data_format="json"):
        self.data_format = data_format

    def deserialize(self, document_path, serialized=None):
        if self.data_format != "json":
            raise NotImplementedError(f"Unsupported data format: {self.data_format}")
        
        if serialized is None:
            return self.__deserialize_document(document_path)
        raw = json.loads(serialized)

        from entities.detection import Detection

        if document_path is Detection:
            return self.__decode_detection_value(value=raw)

        raise ValueError(f"Cannot deserialize into {into}")

    def __deserialize_document(self, source):
        source_path = Path(source)
        if source_path.exists():
            return json.loads(source_path.read_text(encoding="utf-8"))
        return json.loads(source)

    def __decode_detection_value(self, value):
        import torch
        from entities.detection import Detection

        return Detection(score=torch.tensor(value["score"]),
                         label=value["label"],
                         box=torch.tensor(value["box"]))
