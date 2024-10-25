from pathlib import Path
import multiprocessing
import requests

from datasets import load_dataset, Audio, Dataset, DatasetDict


n_cpu = multiprocessing.cpu_count(); n_cpu
data_path = Path.cwd() / "data"
audio_path = data_path / "audio"
audio_path.mkdir(parents=True, exist_ok=True)

dataset_name = "openpecha/tts-training-filtered"
ds = load_dataset(dataset_name)


def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for any HTTP errors
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    except requests.exceptions.RequestException as e:
        raise e

def prepare_audio_file(item):
    file_name = f"{item['file_name']}.wav"
    local_path = audio_path / file_name
    if not local_path.is_file():
        if item["file_name"].startswith("STT_AB"):
            url = f"https://d38pmlk0v88drf.cloudfront.net/AB_wav16k_cleaned/{file_name}"
        else:
            url = item["url"]
        try:
            download_image(url, local_path)
        except:
            local_path = None

    item["path"] = str(local_path)
    del item["file_name"]
    del item["url"]

    return item
    

all_ds = DatasetDict()

all_ds["train"] = ds["train"].map(prepare_audio_file, num_proc=n_cpu-1)
all_ds["test"] = ds["test"].map(prepare_audio_file, num_proc=n_cpu-1)
print(len(all_ds["train"]), len(all_ds["test"]))
all_ds = all_ds.filter(lambda x: x["path"], num_proc=n_cpu)
print(len(all_ds["train"]), len(all_ds["test"]))

dataset_path = data_path / "tts-training-processed"
all_ds.save_to_disk(dataset_path)