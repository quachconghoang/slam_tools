import argparse
import os
import pandas as pd
import shutil
import json
import gc
from pathlib import Path
from config import Config
from pipeline import Pipeline
from reconstruction import reconstruction

def parse_sample_submission(
    base_path: str,
    input_csv: str,
    target_datasets: list,
) -> dict[dict[str, list[Path]]]:
    """Construct a dict describing the test data as 
    
    {"dataset": {"scene": [<image paths>]}}
    """
    data_dict = {}
    with open(input_csv, "r") as f:
        for i, l in enumerate(f):
            # Skip header
            if i == 0:
                print("header:", l)

            if l and i > 0:
                image_path, dataset, scene, _, _ = l.strip().split(',')
                if target_datasets is not None and dataset not in target_datasets:
                    continue

                ### Temp for bug?
                if not os.path.isfile(os.path.join(base_path,image_path)):
                    continue

                if dataset not in data_dict:
                    data_dict[dataset] = {}
                if scene not in data_dict[dataset]:
                    data_dict[dataset][scene] = []
                data_dict[dataset][scene].append(Path(os.path.join(base_path,image_path)))

    for dataset in data_dict:
        for scene in data_dict[dataset]:
            print(f"{dataset} / {scene} -> {len(data_dict[dataset][scene])} images")

    return data_dict


def run(config):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=False)
    args = parser.parse_args()

    if args.output_dir:
        config.output_dir = args.output_dir

    # Check output_dir
    if config.check_exist_dir:
        if os.path.isdir(config.output_dir):
            raise Exception(f"{config.output_dir} is already exists.")

    os.makedirs(config.output_dir, exist_ok=True)
    base_path = os.path.join(config.input_dir_root, "image-matching-challenge-2024")
    feature_dir = os.path.join(config.output_dir, "feature_outputs")
    shutil.copy(config.pipeline_json, config.output_dir)
    shutil.copy(config.transparent_pipeline_json, config.output_dir)

    # Load category
    category_df = pd.read_csv(config.category_csv)
    categories = {}
    for i, row in category_df.iterrows():
        categories[row["scene"]] = row["categories"].split(";")

    data_dict = parse_sample_submission(base_path, config.input_csv, config.target_datasets)
    datasets = list(data_dict.keys())

    submission_path_list = []
    for dataset in datasets:            
        for scene in data_dict[dataset]:
            print(f"[Scene] {scene}")
            work_dir = Path(os.path.join(feature_dir, f"{dataset}_{scene}"))
            work_dir.mkdir(parents=True, exist_ok=True)

            # Switching JSON
            if "transparent" in categories[scene]:
                json_path = config.transparent_pipeline_json
            else:
                json_path = config.pipeline_json
            print(f"catefories: {categories[scene]}")
            print(f"json path: {json_path}")
            
            # Exec Pipeline
            with open(json_path, "r") as f:
                pipeline_config = json.load(f)
            pipeline = Pipeline(data_dict[dataset][scene], work_dir, config.input_dir_root, pipeline_config)
            pipeline.exec()

            # Reconstruction & Save CSV
            print("Start Reconstruction")
            submission_path = reconstruction(data_dict, dataset, scene, base_path, work_dir, config.colmap_mapper_options)
            submission_path_list.append(submission_path)
            gc.collect()
            print("")
    
    # Concat Submission
    submission_df_list = [pd.read_csv(p) for p in submission_path_list]
    submission_df = pd.concat(submission_df_list).reset_index(drop=True)
    submission_df.to_csv(os.path.join(config.output_dir, "submission.csv"), index=False)

if __name__ == '__main__':
    cfg = Config
    run(cfg)
