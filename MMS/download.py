from datasets import load_dataset

dataset_name = "openpecha/tts-sherab"
dataset = load_dataset(dataset_name)
print(dataset)
