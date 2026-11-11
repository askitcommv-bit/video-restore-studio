"""
Persistent per-user settings in config.json (in the project folder): remembered folders,
measured speeds, the Hugging Face token.
"""

import json
from pathlib import Path

from config import CONFIG_JSON_PATH, INPUTS_DIR, OUTPUTS_DIR

def load_user_config() -> dict:
    """Loads persistent user configuration from config.json."""
    if CONFIG_JSON_PATH.exists():
        try:
            with open(CONFIG_JSON_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[CONFIG] Warning: Could not read {CONFIG_JSON_PATH}: {e}")
            return {}
    return {}


def save_user_config(data: dict) -> None:
    """Saves user configuration to config.json."""
    try:
        with open(CONFIG_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[CONFIG] Error: Could not write {CONFIG_JSON_PATH}: {e}")


def get_user_setting(key: str, default: any = None) -> any:
    """Gets a specific setting from user configuration."""
    cfg = load_user_config()
    return cfg.get(key, default)


def set_user_setting(key: str, value: any) -> None:
    """Updates and saves a specific setting in user configuration."""
    cfg = load_user_config()
    cfg[key] = value
    save_user_config(cfg)


def get_last_input_folder() -> Path:
    """Returns the last remembered input folder, defaulting to INPUTS_DIR."""
    val = get_user_setting("last_input_folder")
    if val and Path(val).exists():
        return Path(val)
    return INPUTS_DIR


def set_last_input_folder(folder_path: Path | str) -> None:
    """Remembers the last input folder in persistent config."""
    p = Path(folder_path)
    if p.is_file():
        p = p.parent
    if p.exists():
        set_user_setting("last_input_folder", str(p.resolve()))


def get_last_output_folder() -> Path:
    """Returns the last remembered output folder, defaulting to OUTPUTS_DIR."""
    val = get_user_setting("last_output_folder")
    if val and Path(val).exists():
        return Path(val)
    return OUTPUTS_DIR


def set_last_output_folder(folder_path: Path | str) -> None:
    """Remembers the last output folder in persistent config."""
    p = Path(folder_path)
    if p.is_file():
        p = p.parent
    if p.exists():
        set_user_setting("last_output_folder", str(p.resolve()))
