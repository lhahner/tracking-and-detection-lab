from fastapi import APIRouter 
from services.inference_service import InferenceService 
from api.schemas.inference_request_schema import InferenceRequestSchema

router = APIRouter()

@router.post("/run-inference")
def request_inference_run(request: InferenceRequestSchema):
    InferenceService().parse_dto(request)
    return {"status": "success"} 