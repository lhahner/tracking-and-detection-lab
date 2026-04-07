from util.visualizer import Visualizer
from util.settings_loader import SettingsLoader

class Trainer:
	def __init__(self, settings, dataset):
		self.settings = settings
		self.visualizer =  Visualizer(settings.runtime.datatype)
		self.dataset = dataset