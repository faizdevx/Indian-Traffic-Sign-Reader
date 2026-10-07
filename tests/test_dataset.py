import pandas as pd
import pytest
from PIL import Image

from src.data import (assign_groups, carve_val_from_train, detect_layout, dhash,
                      grouped_stratified_split, list_images, probe_images)
from src.dataset import DataNotPreparedError, TrafficSignDataset, load_prepared
from src.transforms import get_eval_transform
from tests.conftest import write_img


def test_list_images_uses_folder_names_as_labels(folder_dataset):
    df = list_images(folder_dataset)
    assert len(df) == 24
    assert sorted(df.label_name.unique()) == ["class_a", "class_b", "class_c"]
    assert detect_layout(folder_dataset) == "flat"


def test_official_layout_detected(tmp_path):
    for split in ("train", "test"):
        write_img(tmp_path / split / "x" / "1.jpg", 1)
    assert detect_layout(tmp_path) == "official"
    df = list_images(tmp_path)
    assert set(df.official_split) == {"train", "test"}


def test_missing_source_and_empty_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path / "nope")
    with pytest.raises(ValueError):
        list_images(tmp_path)


def test_probe_flags_corrupted_and_hashes_duplicates(folder_dataset):
    (folder_dataset / "class_a" / "bad.jpg").write_bytes(b"not an image")
    (folder_dataset / "class_b" / "copy.jpg").write_bytes((folder_dataset / "class_a" / "0.jpg").read_bytes())
    df = probe_images(folder_dataset, list_images(folder_dataset))
    assert df.corrupted.sum() == 1
    assert df.loc[df.corrupted, "path"].iloc[0].endswith("bad.jpg")
    dup = df[~df.corrupted].groupby("sha256").filter(lambda g: len(g) > 1)
    assert len(dup) == 2


def test_near_duplicates_share_a_group_and_never_straddle_splits(tmp_path):
    root = tmp_path / "d"
    for cls in ("a", "b"):
        for i in range(10):
            write_img(root / cls / f"{i}.png", hash((cls, i)) % 10_000)
            # re-saved lower-quality copy of the same shot = near duplicate
            Image.open(root / cls / f"{i}.png").convert("RGB").save(root / cls / f"{i}_copy.jpg", quality=90)
    df = probe_images(root, list_images(root))
    df["label_id"] = (df.label_name == "b").astype(int)
    df["group"] = assign_groups(df)
    assert df.groupby("group").size().max() >= 2
    df["split"] = grouped_stratified_split(df, seed=0, n_folds=4)
    assert (df.groupby("group")["split"].nunique() == 1).all()


def test_dhash_identical_images_match():
    im = Image.new("RGB", (20, 20), (10, 200, 30))
    assert dhash(im) == dhash(im.copy())


def test_carve_val_keeps_test_untouched():
    n = 60
    df = pd.DataFrame({"official_split": ["train"] * 48 + ["test"] * 12,
                       "label_id": [i % 3 for i in range(n)], "group": range(n)})
    split = carve_val_from_train(df, seed=0)
    assert (split[df.official_split == "test"] == "test").all()
    assert (split == "val").sum() > 0


def test_dataset_item_shape_and_label(folder_dataset):
    df = probe_images(folder_dataset, list_images(folder_dataset))
    df["label_id"] = df.label_name.map({"class_a": 0, "class_b": 1, "class_c": 2})
    ds = TrafficSignDataset(df, folder_dataset, get_eval_transform(64))
    x, y = ds[0]
    assert x.shape == (3, 64, 64) and y == df.label_id.iloc[0]
    assert len(ds) == 24


def test_load_prepared_fails_with_instructions(tmp_path):
    with pytest.raises(DataNotPreparedError, match="prepare_dataset.py"):
        load_prepared(tmp_path / "m.csv", tmp_path / "c.json")


@pytest.mark.integration
def test_real_data_prepared_has_disjoint_splits():
    manifest, names = load_prepared()
    assert not (set(manifest[manifest.split == "train"].sha256) & set(manifest[manifest.split == "test"].sha256))
    assert manifest.label_id.max() == len(names) - 1


def test_smoke_subset_is_a_small_slice_of_the_given_manifest():
    from src.train import smoke_subset
    n = 120
    m = pd.DataFrame({"path": [f"p{i}" for i in range(n)], "label_id": [i % 8 for i in range(n)],
                      "split": ["train"] * 80 + ["val"] * 20 + ["test"] * 20})
    sub = smoke_subset(m, max_classes=3, per_class=2)
    assert set(sub.columns) == set(m.columns)
    assert sub.label_id.nunique() <= 3 and len(sub) <= 3 * 3 * 2
    assert set(sub.path) <= set(m.path)
