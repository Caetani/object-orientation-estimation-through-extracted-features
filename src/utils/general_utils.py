import json
import os
 
import sys
sys.path.insert(0, ".")

def load_config(config_path: str = 'config.json') -> dict:
    """Carrega o arquivo de configuração JSON do projeto."""
    with open(config_path, 'r') as f:
        return json.load(f)


def object_id_to_str(object_id: int) -> str:
    """Converte o id inteiro do objeto para o formato do dataset (ex: 4 -> '000004')."""
    return f'{object_id:06d}'