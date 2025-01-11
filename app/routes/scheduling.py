from fastapi import APIRouter, HTTPException, Query
from app.models.ilp import fetch_data, schedule_classes, post_schedules

router = APIRouter()

@router.get("/generate-schedule")
async def generate_schedule(
    departmentId: int = Query(..., description="Department ID"),
    curriculumId: int = Query(..., description="Curriculum ID"),
    semesterTypeId: int = Query(..., description="Semester Type ID"),
    academicPeriodId: int = Query(..., description="Academic Period ID"),
):
    try:
        # Fetch the required data using dynamic parameters
        params = {
            "departmentId": departmentId,
            "curriculumId": curriculumId,
            "semesterTypeId": semesterTypeId,
            "academicPeriodId": academicPeriodId,
        }
        data = fetch_data(params)  # Pass parameters to fetch_data
        if not data:
            raise HTTPException(status_code=400, detail="Failed to fetch data")

        # Run the scheduling logic
        schedules = schedule_classes(data)
        if not schedules:
            raise HTTPException(status_code=500, detail="Failed to generate schedules")

        # Post the schedules
        post_schedules(schedules)

        return {"status": "success", "message": "Schedules generated and posted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
