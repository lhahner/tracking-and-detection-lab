from registry import MODELS

modes_names = MODELS.names()

class InferenceService:
    def parse_dto(self, inference_request_schema):
        MODELS.register_all(type="detector")
        
    #def get_available_resources():
        