import json
from typing import List, Dict, Any
from torch.utils.data import Subset
from torch.utils import data

# Load dataset
class Dataset(object):
    def __init__(
        self,
        dataset_filepath: str,
    ):
        self.dataset = []
        with open(dataset_filepath, "r") as f:
            for line in f.readlines():
                datapoint = json.loads(line.strip("\n"))
                self.dataset.append(datapoint)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    ############ Format for saving
def convert_dict_of_lists_to_list_of_dicts(dict_of_lists):
    """
    Args:
        dict_of_lists:

    Returns:
        list_ofDict
    """
    list_of_dicts = []
    data = {
        'id': dict_of_lists['id'][0],
        'input':  (dict_of_lists.get('input', [])  or dict_of_lists.get('text', [])  or [None])[0],
        'output': (dict_of_lists.get('output', []) or dict_of_lists.get('label', []) or [None])[0],
        'prediction': dict_of_lists['prediction']
    }
    # if 'answer_choices' in dict_of_lists:
    #     data['answer_choices'] = dict_of_lists['answer_choices'][0]
    list_of_dicts.append(data)
    return list_of_dicts

def collate_fn(batch_of_datapoints: List[Dict]) -> Dict[Any, List]:
    """
    Convert a batch of datapoints into a datapoint that is batched. This is meant to override the default collate function in pytorch and specifically can handle when the value is a list 

    Args:
        batch_ofDatapoints:

    Returns:

    """
    datapoint_batched = {}
    for datapoint in batch_of_datapoints:
        # Gather together all the values per key
        for key, value in datapoint.items():
            if key in datapoint_batched:
                datapoint_batched[key].append(value)
            else:
                datapoint_batched[key] = [value]
    return datapoint_batched


class MultiDataset(data.Dataset):
    def __init__(self, datasets_dict):
        """
        datasets_dict: dict { "task_name": Dataset_obj }
        """
        self.samples = []
        for task, ds in datasets_dict.items():
            for ex in ds:
                # On stocke la donnée + la tâche d'origine
                self.samples.append((task, ex))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        task, sample = self.samples[idx]
        # On enrichit le batch avec la tâche
        sample["task"] = task
        return sample

########### dataset loading
datasets = {
    "truthful_qa": Dataset("datasets/truthful_qa_validation_split.jsonl"),
    "mbpp": Dataset("datasets/mbpp_test_split.jsonl"),
    "gsm8k": Dataset("datasets/gsm8k_test_split.jsonl"),
    "sst-2": Dataset("datasets/imdb_subset.jsonl"),
}

multi_dataset = MultiDataset(datasets)
