"""Execute user Python scripts that build Earth Engine Image / ImageCollection / FeatureCollection graphs."""

_SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "range": range,
    "len": len,
    "float": float,
    "int": int,
    "str": str,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "True": True,
    "False": False,
    "None": None,
}

DEFAULT_IMAGE_FUNCTION_SCRIPT = """\
def apply(image):
    # image: ee.Image — return a transformed ee.Image
    return image

result = apply(image)
"""

DEFAULT_IC_FUNCTION_SCRIPT = """\
def apply(image):
    # image: ee.Image — called once per collection element
    return image

result = image_collection.map(apply)
"""

DEFAULT_FC_FUNCTION_SCRIPT = """\
def apply(feature_collection):
    # feature_collection: ee.FeatureCollection — return a transformed ee.FeatureCollection
    return feature_collection

result = apply(feature_collection)
"""


def _exec_script(script: str, scope: dict):
    import ee

    full_scope = {"ee": ee, "__builtins__": _SAFE_BUILTINS}
    full_scope.update(scope)
    text = (script or "").strip()
    if not text:
        raise ValueError("Script is required.")
    try:
        exec(text, full_scope)  # noqa: S102 — intentional user script hook
    except Exception as exc:
        raise ValueError(f"Script execution failed: {exc}") from exc
    return full_scope


def run_image_script(script: str, image):
    """Run a user script with ``image`` (ee.Image) in scope; return ee.Image."""
    import ee

    scope = _exec_script(script, {"image": image})
    if "result" in scope:
        result = scope["result"]
    elif "apply" in scope and callable(scope["apply"]):
        result = scope["apply"](image)
    else:
        raise ValueError(
            "Script must assign 'result' to an ee.Image, or define "
            "apply(image) and set result = apply(image)."
        )
    if not isinstance(result, ee.Image):
        raise TypeError(
            f"Script result must be ee.Image, got {type(result).__name__}."
        )
    return result


def run_image_collection_script(script: str, image_collection):
    """Run a user script with ``image_collection`` in scope; return ee.ImageCollection."""
    import ee

    scope = _exec_script(script, {"image_collection": image_collection})
    if "result" in scope:
        result = scope["result"]
    elif "apply" in scope and callable(scope["apply"]):
        result = image_collection.map(scope["apply"])
    else:
        raise ValueError(
            "Script must assign 'result' to an ee.ImageCollection, or define "
            "apply(image) and set result = image_collection.map(apply)."
        )
    if not isinstance(result, ee.ImageCollection):
        raise TypeError(
            f"Script result must be ee.ImageCollection, got {type(result).__name__}."
        )
    return result


def run_feature_collection_script(script: str, feature_collection):
    """Run a user script with ``feature_collection`` in scope; return ee.FeatureCollection."""
    import ee

    scope = _exec_script(
        script,
        {"feature_collection": feature_collection, "fc": feature_collection},
    )
    if "result" in scope:
        result = scope["result"]
    elif "apply" in scope and callable(scope["apply"]):
        result = scope["apply"](feature_collection)
    else:
        raise ValueError(
            "Script must assign 'result' to an ee.FeatureCollection, or define "
            "apply(feature_collection) and set result = apply(feature_collection)."
        )
    if not isinstance(result, ee.FeatureCollection):
        raise TypeError(
            f"Script result must be ee.FeatureCollection, got {type(result).__name__}."
        )
    return result
