from abc import ABC, abstractmethod
from dataclasses import dataclass, field

class Metadata(ABC):
    pass

@dataclass(frozen=True)
class NuScenesMetadata(Metadata):
    sample_token: str
    time_stamp: str
    lidar_to_global: list
    ego: list
    aux_info: dict

    def __post_init__(self):
        if self.aux_info is None:
            self.aux_info = {"is_key_frame": True}
