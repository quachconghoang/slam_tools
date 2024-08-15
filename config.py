import os

class Config:
    input_dir_root = "../datas/input"
    output_dir = "../datas/output/exp8/debug"
    check_exist_dir = False
    
    #input_csv = "../datas/input/image-matching-challenge-2024/train/train_submission.csv"
    input_csv = "../datas/input/image-matching-challenge-2024/local_sample_submission.csv"
    category_csv = "../datas/input/image-matching-challenge-2024/train/categories.csv"
    #category_csv = "/mnt/2ndHDD/kaggle/IMC2024/create_dataset/dioscuri/categories.csv"
    #target_datasets = None
    target_datasets = ["church", "lizard", "dioscuri", "multi-temporal-temple-baalshamin", "pond"]
    #target_datasets = ["dioscuri", "multi-temporal-temple-baalshamin"]
    pipeline_json = "exp8/pipeline.json"
    transparent_pipeline_json = "exp8/transp_pipeline.json"
    
    colmap_mapper_options = {
        "min_model_size": 3, # By default colmap does not generate a reconstruction if less than 10 images are registered. Lower it to 3.
        "max_num_models": 2,
        #"num_threads": 1,
    }
