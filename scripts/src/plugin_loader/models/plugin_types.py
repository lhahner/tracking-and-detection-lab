from enum import Enum

class PluginType(Enum):
	DETECTOR = 0
	TRACKER = 1
	DATASETS = 2
	PREPROCESSING = 3
	POST_PROCESSING = 4