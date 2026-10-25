from dataclasses import dataclass

@dataclass
class InferenceRequestSchema:
    detector: str
    pre_processing: list
    post_processing: list
    tracker: str
    dataset: str