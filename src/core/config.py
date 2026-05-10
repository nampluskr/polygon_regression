# src/core/config.py

import os
import re
import yaml
import logging

logger = logging.getLogger("train")


def resolve_vars(config):
    flat = {k: v for k, v in config.items() if isinstance(v, str)}

    def interpolate(value, context):
        # Repeat substitution until no more changes (max 10 to prevent circular refs)
        for _ in range(10):
            new_value = re.sub(
                r'\$\{(\w+)\}',
                lambda m: context.get(m.group(1), m.group(0)),
                value
            )
            if new_value == value:
                break
            value = new_value

        # Raise if any unresolved variables remain
        unresolved = re.findall(r'\$\{(\w+)\}', value)
        if unresolved:
            raise KeyError(f"Undefined variable(s): {unresolved}")
        return value

    # Fully resolve all top-level string variables first
    resolved_flat = {}
    for key in flat:
        resolved_flat[key] = interpolate(flat[key], flat)

    # Apply resolved context to the entire config (including nested dicts)
    resolved = {}
    for k, v in config.items():
        if isinstance(v, str):
            resolved[k] = resolved_flat.get(k, interpolate(v, resolved_flat))
        elif isinstance(v, dict):
            resolved[k] = {
                dk: interpolate(dv, resolved_flat) if isinstance(dv, str) else dv
                for dk, dv in v.items()
            }
        else:
            resolved[k] = v
    return resolved


def load_config(config_path):
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    return resolve_vars(raw)


def merge_configs(default, override):
    merged = default.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged


def save_config(config, config_path):
    save_dir = os.path.dirname(config_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"> config saved: {os.path.basename(config_path)}")