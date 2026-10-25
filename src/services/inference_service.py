from registry import MODELS, TRACKER, DATASETS, POST_PROCESSING, PRE_PROCESSING

modes_names = MODELS.names()

class InferenceService:
    def infer(self, inference_request_schema):
        breakpoint()
    
    def get_available_resources(self):
        MODELS.register_all(type="detector")
        registered_models = MODELS.names()
        
        TRACKER.register_all(type="tracker")
        registered_tracker = TRACKER.names()
        
        DATASETS.register_all(type="datasets")
        registered_dataset = DATASETS.names()
        
        POST_PROCESSING.register_all("post_processing")
        registered_post_processing = POST_PROCESSING.names()
        
        PRE_PROCESSING.register_all("pre_processing")
        registered_pre_processing = PRE_PROCESSING.names()
        return {"models": registered_models,
                "tracker": registered_tracker,
                "datasets": registered_dataset,
                "post_processing": registered_post_processing,
                "pre_processing": registered_pre_processing}
        
        