import pandas as pd
import torch

def load_from_file(filename):
    d = pd.read_csv("data/" + filename, nrows=10000, index_col=False)
    d.index = range(len(d))
    return d

def generate_dataset(raw_data, cols=26):
    input_feature_cols = [f"f{i}" for i in range(cols)]
    input_dataset = raw_data[input_feature_cols]
    output_dataset = raw_data['# label']
    # print(input_dataset.columns)
    # Shape: [10000, 26]
    input_tensors = torch.tensor(input_dataset.values)
    # Shape: [10000]
    output_tensors = torch.tensor(output_dataset.values)
    # A sample is a tuple: (input_tensor, output_tensor)
    dataset = [(input_tensors[i], output_tensors[i]) for i in range(len(raw_data))]
    return dataset

# d = load_from_file("1000_train.csv")
# generate_dataset(d)
