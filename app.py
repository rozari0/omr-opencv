import os
import shutil
import uuid
from typing import List, Optional, Union

from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from main import generate_json_response, scan_omr

app = FastAPI(title="OMR Scanner API", version="1.0.0")


class MCQResponse(BaseModel):
    roll_number: str
    registration: str
    subject_code: str
    mcq_answers: List[Union[str, int]]


class ErrorResponse(BaseModel):
    message: str
    details: Optional[str] = None  # Optional detailed information about the error


@app.post("/", responses={400: {"model": ErrorResponse}, 200: {"model": MCQResponse}})
async def check_file(file: UploadFile):
    if not file.filename.endswith((".jpg", ".jpeg", ".png")):
        error = ErrorResponse(
            message="Invalid file type. Only .jpg, .jpeg, and .png files are allowed.",
            details="Please upload a valid image file.",
        )
        return JSONResponse(
            error.model_dump(),
            status_code=400,
        )
    try:
        file_extension = os.path.splitext(file.filename)[-1]
        random_filename = f"{uuid.uuid4().hex}{file_extension}"
        save_path = os.path.join("uploaded", random_filename)
        os.makedirs("uploaded", exist_ok=True)
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = scan_omr(save_path, save_path + ".png", show=False)
        response = generate_json_response(results)
        print(response)
        return response
    except Exception as e:
        error = ErrorResponse(
            message="An error occurred while processing the file.",
            details=str(e),
        )
        return JSONResponse(
            error.model_dump(),
            status_code=400,
        )
