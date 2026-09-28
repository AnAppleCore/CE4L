import os
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf


def expand_data_root(cfg):
    """Replace the ``<DATA_ROOT>`` placeholder using the ``DATA_ROOT`` environment variable."""
    data_root = os.environ.get("DATA_ROOT", "").rstrip("/")

    def _walk(node) -> None:
        if OmegaConf.is_list(node):
            for index, value in enumerate(list(node)):
                if OmegaConf.is_config(value):
                    _walk(value)
                elif isinstance(value, str) and "<DATA_ROOT>" in value:
                    if not data_root:
                        raise ValueError(
                            "A config path contains <DATA_ROOT>, but DATA_ROOT is not set. "
                            "Example: export DATA_ROOT=/path/to/your/data"
                        )
                    node[index] = value.replace("<DATA_ROOT>", data_root)
            return
        if not OmegaConf.is_dict(node):
            return
        for key in list(node.keys()):
            value = node[key]
            if OmegaConf.is_config(value):
                _walk(value)
            elif isinstance(value, str) and "<DATA_ROOT>" in value:
                if not data_root:
                    raise ValueError(
                        "A config path contains <DATA_ROOT>, but DATA_ROOT is not set. "
                        "Example: export DATA_ROOT=/path/to/your/data"
                    )
                node[key] = value.replace("<DATA_ROOT>", data_root)

    _walk(cfg)
    return cfg


def load_yaml_config(path: str) -> Dict[str, Any]:
    """Load a YAML config file into a plain dict.

    Uses OmegaConf (already used in this repo) to avoid extra deps.
    """
    from omegaconf import OmegaConf

    def _load_with_base(cfg_path: str):
        """Recursively resolve `_base_` chains (string or list)."""
        cfg = OmegaConf.load(cfg_path)

        base_key = "_base_"
        if base_key not in cfg:
            return cfg

        base_val = cfg[base_key]
        # OmegaConf may store lists as ListConfig, so detect via OmegaConf helpers.
        if OmegaConf.is_list(base_val):
            base_list = list(base_val)
        else:
            base_list = [base_val]

        merged = OmegaConf.create({})
        for b in base_list:
            if b is None:
                continue
            b_path = str(b)
            if not os.path.isabs(b_path):
                b_path = os.path.join(os.path.dirname(cfg_path), b_path)
            b_cfg = _load_with_base(b_path)
            merged = OmegaConf.merge(merged, b_cfg)

        # Merge current config last (current overrides base), then drop `_base_`
        merged = OmegaConf.merge(merged, cfg)
        if base_key in merged:
            merged.pop(base_key)
        return merged

    cfg = _load_with_base(path)
    expand_data_root(cfg)

    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def split_argv_config(argv: List[str]) -> Tuple[Optional[str], List[str]]:
    """Extract --config <path> from argv, return (config_path, remaining_argv)."""
    cfg_path = None
    out = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--config" and i + 1 < len(argv):
            cfg_path = argv[i + 1]
            i += 2
            continue
        out.append(a)
        i += 1
    return cfg_path, out


def apply_config_as_defaults(parser, cfg: Dict[str, Any]) -> None:
    """Set argparse defaults using keys present in cfg."""
    # only apply flat key/value pairs
    flat = {k: v for k, v in cfg.items() if isinstance(k, str)}
    parser.set_defaults(**flat)


def inject_required_positionals_if_missing(
    remaining_argv: List[str],
    required_positionals: List[str],
    cfg: Dict[str, Any],
) -> List[str]:
    """If it looks like user didn't provide any positional args, inject them from config.

    Heuristic:
      - if remaining_argv is empty OR first token starts with '-' -> assume no positionals provided.
    """
    if len(required_positionals) == 0:
        return remaining_argv
    if len(remaining_argv) == 0 or remaining_argv[0].startswith("-"):
        injected = []
        for name in required_positionals:
            if name not in cfg:
                raise ValueError(f"Config is missing required positional field: {name}")
            injected.append(str(cfg[name]))
        return injected + remaining_argv
    return remaining_argv


