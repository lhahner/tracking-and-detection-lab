from fastapi import APIRouter 
from services.inference_service import InferenceService 
from api.schemas.inference_request_schema import InferenceRequestSchema

router = APIRouter(prefix="/inference")


@router.get("/available")
def request_available_resources():
    response = InferenceService().get_available_resources()
    return response

@router.post("/run")
def request_inference_run(request: InferenceRequestSchema):
    InferenceService().parse_dto(request)
    return {"status": "success"} 
