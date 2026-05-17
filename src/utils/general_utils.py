import json
import os
 
import sys
sys.path.insert(0, ".")

def load_config(config_path: str = 'config.json') -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)


def object_id_to_str(object_id: int) -> str:
    """ Converts the integer 'object_id' to the dataset format (ex: 4 -> '000004')."""
    return f'{int(object_id):06d}'


def load_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)