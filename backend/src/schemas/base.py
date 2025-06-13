"""Base schema for API responses."""

from pydantic import BaseModel, Field


class ResponseMessage(BaseModel):
    """Base schema for API response messages."""

    message: str = Field(
        ...,
        description="Response message from the API",
    )
