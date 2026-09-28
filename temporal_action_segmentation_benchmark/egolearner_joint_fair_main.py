from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

# Ensure repo root on path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from clego_cl.config_utils import apply_config_as_defaults, load_yaml_config, split_argv_config
from clego_cl.task_map import load_video_to_task
from clego_cl.fair_data import filter_bundle_by_tasks

from model import MultiStageModel
from egolearner_train import Trainer
from egolearner_predict import predict
from egolearner_batch_gen import BatchGenerator
from egobridge_settings import get_annotations_from_settings


# Match continual protocol (fixed task set)
TAS_ALLOWED_TASK_IDS = [1, 3, 4, 5]


def _dump_args_json(output_root: str, args: argparse.Namespace, extra: Dict[str, object]) -> None:
    try:
        payload = {"args": {}, "extra": dict(extra)}
        for k, v in vars(args).items():
            try:
                json.dumps(v)
                payload["args"][k] = v
            except Exception:
                payload["args"][k] = str(v)
        os.makedirs(output_root, exist_ok=True)
        with open(os.path.join(output_root, "args_effective.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("CLEGO TAS joint_fair baseline (joint training with CL-matched data)")
    p.add_argument("--config", type=str, default=None, help="YAML config file (optional). CLI args override config.")

    # fair-joint settings
    p.add_argument("--video_to_task_path", type=str, default=os.path.join(_REPO_ROOT, "video_to_task.npy"))
    p.add_argument("--output_root", type=str, default=None, help="Experiment output directory.")
    p.add_argument("--eval_split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epoch", type=int, default=None, help="For --action=predict/eval: which epoch checkpoint to use (default: num_epochs).")
    p.add_argument("--action", type=str, default="train", choices=["train", "predict"], help="train or predict")

    # baseline args used by model/trainer
    p.add_argument("--feature_path", type=str, default=None)
    p.add_argument("--exp_type", type=str, default="ego-only")
    p.add_argument("--path_data", type=str, default="temporal_action_segmentation_benchmark/tas_annotation")
    p.add_argument("--num_stages", default=4, type=int)
    p.add_argument("--num_layers", default=10, type=int)
    p.add_argument("--num_f_maps", default=64, type=int)
    p.add_argument("--features_dim", default=2048, type=int)
    p.add_argument("--lr", default=0.0005, type=float)
    p.add_argument("--bS", default=1, type=int)
    p.add_argument("--alpha", default=0.15, type=float)
    p.add_argument("--tau", default=4, type=float)
    p.add_argument("--use_target", default="none", choices=["none", "uSv"])
    p.add_argument("--ratio_source", default=1.0, type=float)
    p.add_argument("--ratio_label_source", default=1.0, type=float)
    p.add_argument("--resume_epoch", default=0, type=int)
    p.add_argument("--use_best_model", type=str, default="none", choices=["none", "source", "target"])
    p.add_argument("--multi_gpu", default=False, action="store_true")
    p.add_argument("--verbose", default=False, action="store_true")
    p.add_argument("--use_tensorboard", default=False, action="store_true")
    p.add_argument("--epoch_embedding", default=50, type=int)
    p.add_argument("--stage_embedding", default=-1, type=int)
    p.add_argument("--num_frame_video_embedding", default=50, type=int)

    # Sampling
    p.add_argument("--feat_sample_rate", default=1, type=int)
    p.add_argument("--label_sample_rate", default=2, type=int)
    p.add_argument("--all_sample_rate", default=1, type=int)

    # DA knobs (Trainer.train reads these)
    p.add_argument("--DA_adv", default="none", type=str)
    p.add_argument("--DA_adv_video", default="none", type=str)
    p.add_argument("--pair_ssl", default="all", type=str)
    p.add_argument("--num_seg", default=10, type=int)
    p.add_argument("--place_adv", default=["N", "Y", "Y", "N"], type=str, nargs="+")
    p.add_argument("--multi_adv", default=["N", "N"], type=str, nargs="+")
    p.add_argument("--weighted_domain_loss", default="Y", type=str)
    p.add_argument("--ps_lb", default="soft", type=str)
    p.add_argument("--source_lb_weight", default="pseudo", type=str)
    p.add_argument("--method_centroid", default="none", type=str)
    p.add_argument("--DA_sem", default="mse", type=str)
    p.add_argument("--place_sem", default=["N", "Y", "Y", "N"], type=str, nargs="+")
    p.add_argument("--ratio_ma", default=0.7, type=float)
    p.add_argument("--iter_max_beta", default=[1000, 1000], type=float, nargs="+")
    p.add_argument("--beta", default=[-2, -2], type=float, nargs="+")
    p.add_argument("--gamma", default=-2, type=float)
    p.add_argument("--iter_max_gamma", default=1000, type=float)
    p.add_argument("--DA_ent", default="none", type=str)
    p.add_argument("--place_ent", default=["N", "Y", "Y", "N"], type=str, nargs="+")
    p.add_argument("--mu", default=1, type=float)
    p.add_argument("--use_attn", default="none", choices=["none", "domain_attn"])
    p.add_argument("--DA_dis", default="none", choices=["none", "JAN"])
    p.add_argument("--place_dis", default=["N", "Y", "Y", "N"], type=str, nargs="+")
    p.add_argument("--nu", default=-2, type=float)
    p.add_argument("--iter_max_nu", default=1000, type=float)
    p.add_argument("--DA_ens", default="none", choices=["none", "MCD", "SWD"])
    p.add_argument("--place_ens", default=["N", "Y", "Y", "N"], type=str, nargs="+")
    p.add_argument("--dim_proj", default=128, type=int)
    p.add_argument("--SS_video", default="none", choices=["none", "VCOP"])
    p.add_argument("--place_ss", default=["N", "Y", "Y", "N"], type=str, nargs="+")
    p.add_argument("--eta", default=1, type=float)

    # epochs
    p.add_argument("--num_epochs", default=150, type=int)

    return p


def _filter_bundle_list(
    *,
    bundles: List[str],
    out_dir: str,
    tag: str,
    video_to_task: Dict[str, int],
    allowed_tasks: set[int],
) -> Tuple[List[str], List[Dict[str, object]]]:
    out_paths = []
    meta = []
    for i, b in enumerate(bundles):
        out_path = os.path.join(out_dir, f"{tag}_{i:02d}.bundle")
        st = filter_bundle_by_tasks(in_bundle=b, out_bundle=out_path, video_to_task=video_to_task, allowed_tasks=allowed_tasks)
        out_paths.append(out_path)
        meta.append(
            {
                "in": os.path.abspath(b),
                "out": os.path.abspath(out_path),
                "stats": {
                    "total": st.total,
                    "kept": st.kept,
                    "dropped_not_in_tasks": st.dropped_not_in_tasks,
                    "dropped_unknown_task": st.dropped_unknown_task,
                },
            }
        )
    return out_paths, meta


def main() -> None:
    cfg_path, remaining = split_argv_config(sys.argv[1:])
    parser = build_parser()
    if cfg_path is not None:
        cfg = load_yaml_config(cfg_path)
        apply_config_as_defaults(parser, cfg)
    args = parser.parse_args(remaining)

    if not args.output_root:
        raise ValueError("--output_root must be provided via CLI or --config")
    if not args.feature_path and args.action != "predict":
        raise ValueError("--feature_path must be provided via CLI or --config")

    os.makedirs(args.output_root, exist_ok=True)
    _dump_args_json(args.output_root, args, extra={"entrypoint": "egolearner_joint_fair_main.py", "allowed_tasks": TAS_ALLOWED_TASK_IDS})

    # seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load mapping
    video_to_task = load_video_to_task(args.video_to_task_path)
    allowed_tasks = set(int(x) for x in TAS_ALLOWED_TASK_IDS)

    # resolve split bundles via settings, then filter to allowed task set
    args.dataset = "egobridge"
    args.test = True if args.eval_split == "test" else False
    train_source_bundles, train_source_feat_suffix, val_source_bundles, val_source_feat_suffix, \
        train_target_bundles, train_target_feat_suffix, val_target_bundles, val_target_feat_suffix = get_annotations_from_settings(args)

    # normalize bundle list form
    ts = [train_source_bundles] if isinstance(train_source_bundles, str) else list(train_source_bundles)
    tt = [train_target_bundles] if isinstance(train_target_bundles, str) else list(train_target_bundles)
    vs = [val_source_bundles] if isinstance(val_source_bundles, str) else list(val_source_bundles)
    vt = [val_target_bundles] if isinstance(val_target_bundles, str) else list(val_target_bundles)

    fair_dir = os.path.join(args.output_root, "fair_splits", args.eval_split)
    os.makedirs(fair_dir, exist_ok=True)

    ts_fair, ts_meta = _filter_bundle_list(bundles=ts, out_dir=fair_dir, tag="train_source", video_to_task=video_to_task, allowed_tasks=allowed_tasks)
    tt_fair, tt_meta = _filter_bundle_list(bundles=tt, out_dir=fair_dir, tag="train_target", video_to_task=video_to_task, allowed_tasks=allowed_tasks)
    vs_fair, vs_meta = _filter_bundle_list(bundles=vs, out_dir=fair_dir, tag="eval_source", video_to_task=video_to_task, allowed_tasks=allowed_tasks)
    vt_fair, vt_meta = _filter_bundle_list(bundles=vt, out_dir=fair_dir, tag="eval_target", video_to_task=video_to_task, allowed_tasks=allowed_tasks)

    with open(os.path.join(fair_dir, "fair_split_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "allowed_tasks": sorted(list(allowed_tasks)),
                "exp_type": str(args.exp_type),
                "eval_split": str(args.eval_split),
                "train_source": ts_meta,
                "train_target": tt_meta,
                "eval_source": vs_meta,
                "eval_target": vt_meta,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # directories
    model_dir = os.path.join(args.output_root, "models")
    results_dir = os.path.join(args.output_root, "results", args.eval_split)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    actions_dict = {str(i): i for i in range(28)}  # 1 + 27
    num_classes = len(actions_dict)
    model = MultiStageModel(args, num_classes)
    trainer = Trainer(num_classes)

    gt_path = os.path.join(args.path_data, "gts_fps25/")

    if args.action == "train":
        # build loaders (reading only fair bundles)
        batch_gen_source_train = BatchGenerator(
            num_classes, actions_dict, gt_path, args.feature_path,
            feat_sample_rate=args.feat_sample_rate, label_sample_rate=args.label_sample_rate,
            all_sample_rate=args.all_sample_rate, feat_suffix=train_source_feat_suffix,
            video_to_task=video_to_task,
        )
        batch_gen_target_train = BatchGenerator(
            num_classes, actions_dict, gt_path, args.feature_path,
            feat_sample_rate=args.feat_sample_rate, label_sample_rate=args.label_sample_rate,
            all_sample_rate=args.all_sample_rate, feat_suffix=train_target_feat_suffix,
            video_to_task=video_to_task,
        )
        batch_gen_source_val = BatchGenerator(
            num_classes, actions_dict, gt_path, args.feature_path,
            feat_sample_rate=args.feat_sample_rate, label_sample_rate=args.label_sample_rate,
            all_sample_rate=args.all_sample_rate, feat_suffix=val_source_feat_suffix,
            video_to_task=video_to_task,
        )
        batch_gen_target_val = BatchGenerator(
            num_classes, actions_dict, gt_path, args.feature_path,
            feat_sample_rate=args.feat_sample_rate, label_sample_rate=args.label_sample_rate,
            all_sample_rate=args.all_sample_rate, feat_suffix=val_target_feat_suffix,
            video_to_task=video_to_task,
        )
        batch_gen_source_train.read_data(ts_fair)
        batch_gen_target_train.read_data(tt_fair)
        batch_gen_source_val.read_data(vs_fair)
        batch_gen_target_val.read_data(vt_fair)

        start = time.time()
        trainer.train(
            model,
            model_dir,
            results_dir,
            batch_gen_source_train,
            batch_gen_target_train,
            batch_gen_source_val,
            batch_gen_target_val,
            device,
            args,
        )
        if args.verbose:
            print(f"[joint_fair] training time sec: {time.time() - start:.1f}")
        return

    # predict
    epoch = int(args.num_epochs) if args.epoch is None else int(args.epoch)
    predict(
        model=model,
        model_dir=model_dir,
        results_dir=results_dir,
        features_path=args.feature_path,
        vid_list_file=vt_fair,
        feat_suffix=val_target_feat_suffix,
        feat_sample_rate=args.feat_sample_rate,
        all_sample_rate=args.all_sample_rate,
        epoch=epoch,
        actions_dict=actions_dict,
        device=device,
        args=args,
    )


if __name__ == "__main__":
    main()


