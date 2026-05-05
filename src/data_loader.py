import pandas as pd

def load_data():
    df = pd.read_csv("data/train_with_task_type.csv")
    df = df.drop_duplicates(subset=["prompt"])
    print("Dataset size after dedup:", len(df))
    X = df["prompt"]
    y = df["task_type"]
    return X, y