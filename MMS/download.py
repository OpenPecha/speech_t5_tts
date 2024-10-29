from datasets import load_dataset

dataset_name = "openpecha/tts-sherab-grade3"
dataset = load_dataset(dataset_name)
print(dataset)
