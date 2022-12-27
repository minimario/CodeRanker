from datasets import (
    load_dataset,
    concatenate_datasets,
    DatasetDict,
)
import numpy as np

# data_files = {"train": "train.json", "val": "val.json"}
# dataset = load_dataset("json", data_files={"train": "train.json", "val":"val.json"}, cache_dir="./cache")

def get_partials(code, n=3):
    code = code.split("\n")
    num_lines = len(code)
    partial_programs = []
    proportions = [i/(n+1) for i in range(1, n+1)]
    for i in proportions:
        partial_line_count = max(1, int(num_lines*i))
        partial_programs.append('\n'.join(code[0:partial_line_count]))
    return partial_programs


def create_extended_dataset(dataset, n=3):
    # Given a dataset, creates a new dataset that contains
    # the original dataset and the dataset with the 
    # first 25%, 50%, and 75% of each code snippet
    dataset_new = DatasetDict()
    for split in dataset:
        dataset_split = dataset[split]
        partials = [get_partials(completion) for completion in dataset_split["completion"]]

        to_concatenate = [dataset_split]
        for i in range(n):
            def add_partial(example, idx):
                return {"completion": partials[idx][i]} 
            dataset_mapped = dataset_split.map(add_partial, with_indices=True)
            to_concatenate.append(dataset_mapped)
        
        dataset_new[split] = dataset_concatenate = concatenate_datasets(to_concatenate)
    
    print("Sizes of new dataset:")
    for split in dataset_new:
        print(split, len(dataset_new[split]))
    return dataset_new

def create_grouped_indices(val_grouped_indices, val_grouped_labels, n=3):
    # val_grouped_indices = np.load('val_grouped_indices.npy', allow_pickle=True)
    # val_grouped_labels = np.load('val_grouped_labels.npy', allow_pickle=True)

    # assert 57246 == np.count_nonzero(val_grouped_indices != -1)
    dataset_size = np.count_nonzero(val_grouped_indices != -1)
    print("Total val dataset size: ", (n+1) * dataset_size)
    
    to_concatenate = [val_grouped_indices]
    for i in range(n):
        to_concatenate.append(np.where(val_grouped_indices != -1, val_grouped_indices + dataset_size * (i+1), val_grouped_indices))
    val_grouped_indices_new = np.concatenate(to_concatenate, axis=1)
    val_grouped_labels_new = np.concatenate([val_grouped_labels] * (n+1), axis=1)

    print("Shape of new val_grouped_indices: ", val_grouped_indices_new.shape)

    # np.save('val_grouped_indices_new.npy', val_grouped_indices_new)
    # np.save('val_grouped_labels_new.npy', val_grouped_labels_new)

    return val_grouped_indices_new, val_grouped_labels_new