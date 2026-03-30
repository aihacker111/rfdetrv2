# ------------------------------------------------------------------------
# Small helpers for the high-level DETR API package.
# ------------------------------------------------------------------------


def pydantic_dump(obj):
    return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()
