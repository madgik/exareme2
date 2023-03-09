from pydantic import BaseModel


class ImmutableBaseModel(BaseModel):
    class Config:
        allow_mutation = False
