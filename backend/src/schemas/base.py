from pydantic import BaseModel, Field

class ResponseMessage(BaseModel):
    message: str = Field(
        ...,
        description="Response message from the API",
    )


