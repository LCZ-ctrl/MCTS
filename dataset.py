from torch.utils.data import Dataset
from datasets import load_dataset

class TaskDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = ex["question"] + " 答案："
        enc = self.tokenizer(
            prompt, max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids": enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "label": ex["answer"]
        }

def load_task_dataset(task_name, dataset_name, data_dir, tokenizer):
    if task_name == "mathqa":
        ds = load_dataset("openai/grade-school-math", cache_dir=f"{data_dir}/GSM8K")
    elif task_name == "csr":
        ds = load_dataset("ai2_arc", "ARC-Challenge", cache_dir=f"{data_dir}/ARC-C")
    else:
        raise ValueError(f"Unsupported task: {task_name}")
    examples = [{"question": x["question"], "answer": x["answer"][0]} for x in ds["train"]]
    return TaskDataset(examples, tokenizer)
