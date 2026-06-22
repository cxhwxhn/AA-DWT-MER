import argparse
import json
import os
import random
import sys

import numpy as np
import torch

from data_utils import (
    MicroExpressionDataset,
    build_label_map,
    group_rows_by_dataset,
    normalize_dataset_name,
    read_manifest,
)
from model import AADWTMER
from train import calculate_metrics, init_confusion_matrix, train


PREDEFINED_LOSO = {
    "SMIC": {
        3: ["s1", "s2", "s3", "s4", "s5", "s6", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s18", "s19", "s20"],
    },
    "CASME": {
        4: ["sub01", "sub02", "sub03", "sub04", "sub05", "sub06", "sub07", "sub08", "sub09", "sub10", "sub11", "sub12", "sub13", "sub14", "sub15", "sub16", "sub17", "sub18", "sub19"],
    },
    "CASMEII": {
        5: [ "sub01","sub02", "sub03", "sub04", "sub05", "sub06", "sub07", "sub08", "sub09", "sub10", "sub11", "sub12", "sub13", "sub14", "sub15", "sub16", "sub17", "sub18", "sub19", "sub20", "sub21", "sub22", "sub23", "sub24", "sub25", "sub26"],
    },
}

# PREDEFINED_LOSO = {
    
# }


CLASS_PROTOCOLS = {
    "SMIC": {
        "num_classes": 3,
        "labels": ["positive", "negative", "surprise"],
    },
    "CASME": {
        "num_classes": 4,
        "labels": ["disgust", "surprise", "repression", "tense"],
    },
    "CASMEII": {
        "num_classes": 5,
        "labels": ["happiness", "disgust", "surprise", "repression", "others"],
    },
}


def set_random_seed(SEED=2023, use_cudnn=False):
    random.seed(SEED)
    np.random.seed(np.int64(SEED))
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_loso_subjects(dataset_rows):
    return sorted({row["subject"] for row in dataset_rows})


def get_loso_list(args, dataset_name, dataset_rows):
    manifest_subjects = set(get_loso_subjects(dataset_rows))
    predefined_loso = PREDEFINED_LOSO.get(dataset_name, {}).get(args.cls)
    if predefined_loso is None:
        loso = sorted(manifest_subjects)
        print("LOSO list is generated from manifest for {}: {}".format(dataset_name, loso))
        return loso

    loso = [subject for subject in predefined_loso if subject in manifest_subjects]
    missing_subjects = [subject for subject in predefined_loso if subject not in manifest_subjects]
    extra_subjects = sorted(manifest_subjects - set(predefined_loso))

    if missing_subjects:
        print("Warning: {} predefined LOSO subjects not found in manifest: {}".format(dataset_name, missing_subjects))
    if extra_subjects:
        print("Warning: {} manifest has extra subjects not in predefined LOSO: {}".format(dataset_name, extra_subjects))

    if not loso:
        loso = sorted(manifest_subjects)
        print("Warning: predefined LOSO does not match manifest subjects. Use manifest LOSO instead: {}".format(loso))
        return loso

    print("LOSO list for {}: {}".format(dataset_name, loso))
    return loso


def build_subject_dataset(rows, label_to_idx, args, train_mode):
    aug_config = {
        "crop_scale": args.aug_crop_scale,
        "rotation": args.aug_rotation,
        "brightness": args.aug_brightness,
        "contrast": args.aug_contrast,
        "saturation": args.aug_saturation,
        "grayscale_p": args.aug_grayscale_p,
        "horizontal_flip_p": args.aug_hflip_p,
    }
    return MicroExpressionDataset(
        rows,
        label_to_idx=label_to_idx,
        image_size=args.image_size,
        train=train_mode,
        aug_config=aug_config,
    )


def print_args(args):
    print("----------args----------")
    for k in list(vars(args).keys()):
        print("%s: %s" % (k, vars(args)[k]))
    print("----------args----------")


def build_protocol_label_map(dataset_name, dataset_rows):
    protocol = CLASS_PROTOCOLS.get(dataset_name)
    if protocol is None:
        return build_label_map(dataset_rows)

    expected_labels = set(protocol["labels"])
    actual_labels = {row["label"] for row in dataset_rows}
    if actual_labels != expected_labels:
        raise ValueError(
            "{} labels should be {}, but manifest has {}".format(
                dataset_name, sorted(expected_labels), sorted(actual_labels)
            )
        )
    return {label: index for index, label in enumerate(protocol["labels"])}


def run_one_dataset(args, dataset_name, dataset_rows):
    label_to_idx = build_protocol_label_map(dataset_name, dataset_rows)
    args.cls = len(label_to_idx)

    loso = get_loso_list(args, dataset_name, dataset_rows)
    if len(loso) < 2:
        raise ValueError(
            "{} has {} subject(s). LOSO training needs at least two subjects.".format(
                dataset_name, len(loso)
            )
        )
    confusion_matrix = init_confusion_matrix(args.cls)

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, dataset_name + "_label_map.json"), "w", encoding="utf-8") as file:
        json.dump(label_to_idx, file, indent=2, ensure_ascii=False)

    print_args(args)

    print(loso)
    for sub in range(len(loso)):
        subject = loso[sub]
        test_rows = [row for row in dataset_rows if row["subject"] == subject]
        train_rows = [row for row in dataset_rows if row["subject"] != subject]

        train_dataset = build_subject_dataset(train_rows, label_to_idx, args, train_mode=True)
        test_dataset = build_subject_dataset(test_rows, label_to_idx, args, train_mode=False)

        model = AADWTMER(
            num_classes=args.cls,
            pretrained=not args.no_pretrained,
            groups=args.groups,
            dropout=args.dropout,
        )

        if args.pretrained_path is not None:
            print("loading model.....")
            pretrain_weight = torch.load(args.pretrained_path)
            model.load_state_dict(pretrain_weight, strict=False)

        args.save_path = os.path.join(
            args.save_dir,
            args.version + "_" + dataset_name + "_" + subject + "_" + str(args.cls) + "cls.pth",
        )

        print("LOSO {}".format(subject))
       

        final_acc, subject_confusion_matrix = train(
            args=args,
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )
        print("LOSO {} best_acc:{}".format(subject, final_acc))

        for i in range(len(confusion_matrix)):
            for j in range(len(confusion_matrix[0])):
                confusion_matrix[i][j] += subject_confusion_matrix[i][j]

    acc, war, uar, wf1, uf1 = calculate_metrics(confusion_matrix)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", nargs="+", default=["data/CASMEII/CASME2.xlsx"])
    parser.add_argument("--datasets", nargs="*", default=["CASMEII"])
    parser.add_argument("--frame_window", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--eval_interval", type=int, default=1, help="Evaluate every N training epochs")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--grad_clip", type=float, default=1)
    parser.add_argument("--save_dir", default="saved_models")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--pretrained_path", default=None)
    parser.add_argument("--version", default="V1.0.0")
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--aug_crop_scale", type=float, default=0.95)
    parser.add_argument("--aug_rotation", type=float, default=1.0)
    parser.add_argument("--aug_brightness", type=float, default=0.02)
    parser.add_argument("--aug_contrast", type=float, default=0.05)
    parser.add_argument("--aug_saturation", type=float, default=0.0)
    parser.add_argument("--aug_grayscale_p", type=float, default=0.0)
    parser.add_argument("--aug_hflip_p", type=float, default=0.0)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=10)
    parser.add_argument("--backbone_lr_scale", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--use_cudnn",
        action="store_true",
        help="Enable cuDNN. Disabled by default to avoid convolution algorithm errors.",
    )
    parser.add_argument(
        "--no_pretrained",
        action="store_true",
        default=False,
        help="Do not load ImageNet pretrained ResNet18 weights.",
    )
    parser.add_argument(
        "--use_pretrained",
        action="store_false",
        dest="no_pretrained",
        help="Load ImageNet pretrained ResNet18 weights. This is the default.",
    )
    args = parser.parse_args()

    print("========== LOSO training started ==========")
    print("python:", sys.executable)
    print("manifest:", args.manifest)
    print("version:", args.version)

    set_random_seed(args.seed, use_cudnn=args.use_cudnn)
    rows = read_manifest(args.manifest, frame_window=args.frame_window)
    print("loaded samples:", len(rows))
    rows_by_dataset = group_rows_by_dataset(rows)
    print("loaded datasets:", list(rows_by_dataset.keys()))

    if args.datasets is not None:
        requested_datasets = {normalize_dataset_name(name) for name in args.datasets}
        rows_by_dataset = {
            dataset_name: dataset_rows
            for dataset_name, dataset_rows in rows_by_dataset.items()
            if dataset_name in requested_datasets
        }
        missing_datasets = requested_datasets - set(rows_by_dataset)
        if missing_datasets:
            raise ValueError("Datasets not found in manifest: {}".format(sorted(missing_datasets)))

    if not rows_by_dataset:
        raise ValueError("No dataset rows found for training.")

    for dataset_name, dataset_rows in rows_by_dataset.items():
        run_one_dataset(args, dataset_name, dataset_rows)
