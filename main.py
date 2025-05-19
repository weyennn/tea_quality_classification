import os
from src.build_dataset import build_dataset
import shutil
import random
import pandas as pd

def split_sensor_files(source_dir="data", output_dir="output", split_ratio=0.8, seed=42):
    random.seed(seed)
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for label in os.listdir(source_dir):
        class_path = os.path.join(source_dir, label)
        if not os.path.isdir(class_path):
            continue

        all_files = [f for f in os.listdir(class_path) if f.endswith(".csv")]
        random.shuffle(all_files)
        split_point = int(len(all_files) * split_ratio)

        train_files = all_files[:split_point]
        test_files = all_files[split_point:]

        for f in train_files:
            src = os.path.join(class_path, f)
            dst = os.path.join(train_dir, label, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

        for f in test_files:
            src = os.path.join(class_path, f)
            dst = os.path.join(test_dir, label, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

    print("Splitting selesai.")
    print(f" - Train folder: {train_dir}")
    print(f" - Test folder : {test_dir}")

def main():
    print("[STEP 1] Splitting data sensor mentah...")
    split_sensor_files()

    print("[STEP 2] Ekstraksi fitur dari train dan test...")
    df_train = build_dataset(base_path="output/train")
    df_test = build_dataset(base_path="output/test")

    print("[STEP 3] Menyimpan dataset hasil ekstraksi...")
    df_train.to_csv("output/train_dataset.csv", index=False)
    df_test.to_csv("output/test_dataset.csv", index=False)

    print("[DONE] Dataset siap digunakan!")
    print(" - train_dataset.csv")
    print(" - test_dataset.csv")

if __name__ == "__main__":
    main()
