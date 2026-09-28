
import os
import os.path as osp

from omegaconf import OmegaConf


def _expand_data_root(cfg):
    """Replace ``<DATA_ROOT>`` using the DATA_ROOT environment variable."""
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


def load_config(cfg_file):
    cfg = OmegaConf.load(cfg_file)
    if '_base_' in cfg:
        if isinstance(cfg._base_, str):
            # Support recursive base configs (a base config can itself define _base_).
            base_cfg = load_config(osp.join(osp.dirname(cfg_file), cfg._base_))
        else:
            # If multiple bases are provided, resolve each (recursively) then merge.
            base_cfg = OmegaConf.merge(load_config(osp.join(osp.dirname(cfg_file), f)) for f in cfg._base_)
        cfg = OmegaConf.merge(base_cfg, cfg)
    _expand_data_root(cfg)
    return cfg

def get_config(args):
    cfg = load_config(args.config)

    # Apply optional dotlist overrides before freezing the config.
    # This enables simple CLI overrides like:
    #   --opts train.seed=0 output=./exps/joint_fair/train_seed0/
    if hasattr(args, "opts") and args.opts:
        # Temporarily allow updates; dotlist must refer to existing keys unless the base config defines them.
        try:
            override = OmegaConf.from_dotlist(list(args.opts))
            cfg = OmegaConf.merge(cfg, override)
        except Exception as e:
            raise ValueError(f"Failed to apply --opts overrides: {args.opts}. Error: {e}")

    OmegaConf.set_struct(cfg, True)

    if hasattr(args, 'output') and args.output:
        cfg.output = args.output

    cfg.local_rank = args.local_rank

    OmegaConf.set_readonly(cfg, True)

    return cfg
