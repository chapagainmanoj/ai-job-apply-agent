from typing import Literal
import json


def extract_json_from_response(response: str, type: Literal["object", "list"] = "object"):
    try:
        if type == "object":
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            json_str = response[json_start:json_end]

            parsed_data = json.loads(json_str)
            return parsed_data
        elif type == "list":
            json_str = response[response.find("[") : response.rfind("]") + 1]
            return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Claude response: {response}")
        raise ValueError(f"Failed to parse Claude response as JSON: {e}")
