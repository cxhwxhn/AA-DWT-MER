import csv
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MicroExpressionDataset(Dataset):
    def __init__(self, rows, label_to_idx, image_size=224, train=True, aug_config=None):
        self.rows = rows
        self.label_to_idx = label_to_idx
        self.transform = build_transform(image_size, train, aug_config)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        label = self.label_to_idx[row["label"]]
        return self.transform(image), torch.tensor(label, dtype=torch.long)


def build_transform(image_size, train, aug_config=None):
    if train:
        aug_config = aug_config or {}
        crop_scale = aug_config.get("crop_scale", 0.9)
        rotation = aug_config.get("rotation", 5)
        brightness = aug_config.get("brightness", 0.1)
        contrast = aug_config.get("contrast", 0.15)
        saturation = aug_config.get("saturation", 0.05)
        grayscale_p = aug_config.get("grayscale_p", 0.0)
        horizontal_flip_p = aug_config.get("horizontal_flip_p", 0.5)

        return transforms.Compose(
            [
                transforms.Resize((image_size + 24, image_size + 24)),
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(crop_scale, 1.0),
                    ratio=(0.95, 1.05),
                ),
                transforms.RandomHorizontalFlip(p=horizontal_flip_p),
                transforms.RandomRotation(degrees=rotation),
                transforms.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                ),
                transforms.RandomGrayscale(p=grayscale_p),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def read_manifest(csv_paths, frame_window=2):
    rows = []
    required = {"dataset", "subject", "label", "image_path"}
    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if csv_path.suffix.lower() == ".xlsx":
            rows.extend(read_casmeii_coding_xlsx(csv_path, frame_window=frame_window))
            continue

        base_dir = csv_path.parent
        with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

            for row in reader:
                image_path = Path(row["image_path"])
                if not image_path.is_absolute():
                    image_path = base_dir / image_path
                if not image_path.exists():
                    raise FileNotFoundError(f"Image not found: {image_path}")

                rows.append(
                    {
                        "dataset": normalize_dataset_name(row["dataset"]),
                        "subject": row["subject"].strip(),
                        "label": row["label"].strip().lower(),
                        "image_path": str(image_path),
                    }
                )
    if not rows:
        raise ValueError("No samples were found in the manifest files.")
    return rows


def read_casmeii_coding_xlsx(xlsx_path, frame_window=2):
    image_root = xlsx_path.parent
    if (image_root / "CASMEII").is_dir():
        image_root = image_root / "CASMEII"
    shared_strings, sheet_xml = _load_xlsx_sheet(xlsx_path)
    rows = []
    header = {}

    for row_index, row_values in enumerate(_iter_xlsx_rows(sheet_xml, shared_strings), start=1):
        if row_index == 1:
            header = {
                value.strip(): column_index
                for column_index, value in enumerate(row_values)
                if value.strip()
            }
            required = {"Subject", "Filename", "ApexFrame", "Estimated Emotion"}
            missing = required - set(header)
            if missing:
                raise ValueError(f"{xlsx_path} missing columns: {sorted(missing)}")
            continue

        subject_raw = _xlsx_value(row_values, header["Subject"])
        filename = _xlsx_value(row_values, header["Filename"])
        apex_frame = _xlsx_value(row_values, header["ApexFrame"])
        emotion = _xlsx_value(row_values, header["Estimated Emotion"]).lower()
        if not subject_raw or not filename or not apex_frame or not emotion:
            continue

        subject = f"sub{int(float(subject_raw)):02d}"
        label = _map_casmeii_label(emotion)
        try:
            apex_number = int(float(apex_frame))
        except ValueError as error:
            print(f"Warning: skip CASMEII row {row_index}: {error}")
            continue

        image_paths = _find_casmeii_window_images(
            image_root=image_root,
            subject=subject,
            sequence=filename,
            apex_number=apex_number,
            frame_window=frame_window,
        )
        if not image_paths:
            print(
                "Warning: skip CASMEII row {}: no image found around apex {}".format(
                    row_index, apex_number
                )
            )
            continue

        for image_path in image_paths:
            rows.append(
                {
                    "dataset": "CASMEII",
                    "subject": subject,
                    "label": label,
                    "image_path": str(image_path),
                }
            )

    if not rows:
        raise ValueError(f"No CASMEII samples were generated from {xlsx_path}.")
    return rows


def _load_xlsx_sheet(xlsx_path):
    namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(xlsx_path) as archive:
        shared_strings = []
        shared_xml = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        for item in shared_xml.findall("main:si", namespace):
            text_parts = [node.text or "" for node in item.findall(".//main:t", namespace)]
            shared_strings.append("".join(text_parts))

        sheet_xml = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    return shared_strings, sheet_xml


def _iter_xlsx_rows(sheet_xml, shared_strings):
    namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    for row in sheet_xml.findall(".//main:sheetData/main:row", namespace):
        values = []
        for cell in row.findall("main:c", namespace):
            column_index = _column_index(cell.attrib["r"])
            while len(values) <= column_index:
                values.append("")
            value_node = cell.find("main:v", namespace)
            if value_node is None:
                values[column_index] = ""
            elif cell.attrib.get("t") == "s":
                values[column_index] = shared_strings[int(value_node.text)]
            else:
                values[column_index] = value_node.text or ""
        yield values


def _column_index(cell_ref):
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter.upper()) - ord("A") + 1
    return index - 1


def _xlsx_value(row_values, index):
    if index >= len(row_values):
        return ""
    return row_values[index].strip()


def _map_casmeii_label(emotion):
    if emotion in {"happiness", "disgust", "surprise", "repression"}:
        return emotion
    return "others"


def normalize_dataset_name(dataset_name):
    dataset_name = dataset_name.strip()
    dataset_key = dataset_name.upper().replace("-", "").replace("_", "")
    aliases = {
        "SMIC": "SMIC",
        "CASME": "CASME",
        "CASMEII": "CASMEII",
        "CASME2": "CASMEII",
    }
    return aliases.get(dataset_key, dataset_name)


def _find_casmeii_apex_image(image_root, subject, sequence, apex_number):
    sequence_dir = image_root / subject / sequence
    for extension in (".jpg", ".jpeg", ".png"):
        for prefix in ("reg_img", "img"):
            image_path = sequence_dir / f"{prefix}{apex_number}{extension}"
            if image_path.exists():
                return image_path
    raise FileNotFoundError(
        f"CASMEII apex image not found: {sequence_dir / ('reg_img' + str(apex_number) + '.jpg')} or {sequence_dir / ('img' + str(apex_number) + '.jpg')}"
    )


def _find_casmeii_window_images(image_root, subject, sequence, apex_number, frame_window):
    image_paths = []
    start_frame = max(1, apex_number - frame_window)
    end_frame = apex_number + frame_window
    for frame_number in range(start_frame, end_frame + 1):
        try:
            image_paths.append(
                _find_casmeii_apex_image(
                    image_root=image_root,
                    subject=subject,
                    sequence=sequence,
                    apex_number=frame_number,
                )
            )
        except FileNotFoundError:
            continue
    return image_paths


def group_rows_by_dataset(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    return dict(sorted(grouped.items()))


def build_label_map(rows):
    labels = sorted({row["label"] for row in rows})
    return {label: index for index, label in enumerate(labels)}


def build_loso_splits_for_dataset(rows):
    subjects = sorted({row["subject"] for row in rows})
    for subject in subjects:
        train_rows = [row for row in rows if row["subject"] != subject]
        test_rows = [row for row in rows if row["subject"] == subject]
        yield subject, train_rows, test_rows
