#!/usr/bin/env python
# coding: utf-8

# In[66]:


from pathlib import Path
import multiprocessing
import requests

from datasets import load_dataset, Audio, Dataset, DatasetDict

import multiprocessing
from pathlib import Path
from collections import Counter

from transformers import SpeechT5Processor, pipeline
from datasets import DatasetDict, Audio, concatenate_datasets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio as IAudio
from IPython.display import display

from phonetic import text2phon


# In[45]:


n_cpu = multiprocessing.cpu_count()-5; print("CPU count:", n_cpu)
data_path = Path("/data/volume")
audio_path = data_path / "audio"
audio_path.mkdir(parents=True, exist_ok=True)


# In[46]:


processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")


# ## Load the dataset

# In[47]:


dataset_name = "openpecha/tts-training-filtered"
dataset = load_dataset(dataset_name)
dataset


# In[48]:


dataset["train"][100]["uni"]


# ## Filter Dataset
# 
# ### Ignore Long Audio
# Some of the examples in the dataset are apparently longer than the maximum input length the model can handle (600 tokens), so we should remove those from the dataset. In fact, to allow for larger batch sizes we'll remove anything over 200 tokens.

# In[49]:


dataset = concatenate_datasets([dataset["train"], dataset["test"]])


# In[50]:


def compute_tokens_len(item):
    tokens = processor.tokenizer.tokenize(item["sentence"])
    item["tokens_len"] = len(tokens)
    return item

dataset = dataset.map(compute_tokens_len, num_proc=n_cpu)


# In[51]:


dataset[0]["tokens_len"]


# In[52]:


def plot_token_len_histogram(token_lengths):
    # Plotting the histogram
    plt.figure(figsize=(10, 6))  # Set the figure size
    plt.hist(token_lengths, bins=20, edgecolor='black')  # Create a histogram with 10 bins

    # Adding titles and labels
    plt.title('Histogram of Token Lengths')
    plt.xlabel('Token Length')
    plt.ylabel('Frequency')

    # Show the chart
    plt.show()


# In[53]:


# plot_token_len_histogram(dataset["tokens_len"])


# In[54]:


dataset = dataset.filter(lambda x: x["tokens_len"]<200, num_proc=n_cpu)
len(dataset)


# In[56]:


# plot_token_len_histogram(dataset["tokens_len"])


# ### Balance out Departments

# In[57]:


def plot_department_dist(dataset):
    c = Counter(dataset["label"])
    print(c)
    labels = list(c.keys())
    sizes =  list(c.values()) # Percentages or absolute values
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', "#ffff99"]  # Optional colors

    # Create the pie chart
    plt.figure(figsize=(4, 4))  # Set the figure size
    plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',  # Show percentages with 1 decimal
        shadow=True,        # Add shadow effect
        startangle=140      # Rotate the start angle
    )

    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal')

    # Show the chart
    plt.title('Department')
    plt.show()


# In[58]:


# plot_department_dist(dataset)


# In[59]:


def get_balance_dataset(dataset, total, seed=42):
    ab_dataset = dataset.filter(lambda x: x["label"] == "STT_AB", num_proc=n_cpu).shuffle(seed=seed)
    nw_dataset = dataset.filter(lambda x: x["label"] == "STT_NW", num_proc=n_cpu).shuffle(seed=seed)
    hs_dataset = dataset.filter(lambda x: x["label"] == "STT_HS", num_proc=n_cpu).shuffle(seed=seed)
    pc_dataset = dataset.filter(lambda x: x["label"] == "STT_PC", num_proc=n_cpu).shuffle(seed=seed)
    ns_dataset = dataset.filter(lambda x: x["label"] == "STT_NS", num_proc=n_cpu).shuffle(seed=seed)

    ab_dataset = ab_dataset.select(range(total//5))
    nw_dataset = nw_dataset.select(range(total//5))
    hs_dataset = hs_dataset.select(range(total//5))
    pc_dataset = pc_dataset.select(range(total//5))
    ns_dataset = ns_dataset.select(range(total//5))

    ab_dataset = ab_dataset.train_test_split(test_size=0.2, seed=42)
    nw_dataset = nw_dataset.train_test_split(test_size=0.2, seed=42)
    hs_dataset = hs_dataset.train_test_split(test_size=0.2, seed=42)
    pc_dataset = pc_dataset.train_test_split(test_size=0.2, seed=42)
    ns_dataset = ns_dataset.train_test_split(test_size=0.2, seed=42)


    dataset = DatasetDict({
        "train": concatenate_datasets([ab_dataset["train"], nw_dataset["train"], hs_dataset["train"], pc_dataset["train"], ns_dataset["train"]]),
        "test": concatenate_datasets([ab_dataset["test"], nw_dataset["test"], hs_dataset["test"], pc_dataset["test"], ns_dataset["test"]])
    })

    return dataset


# In[69]:


balanced_dataset = get_balance_dataset(dataset, total=300000)
len(balanced_dataset)


# In[61]:


# plot_department_dist(concatenate_datasets([balanced_dataset["train"], balanced_dataset["test"]]))


# In[62]:


dataset = balanced_dataset


# ## Set Audio file path

# In[63]:


def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for any HTTP errors
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    except requests.exceptions.RequestException as e:
        raise e

def resolve_audio_path(item):
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

    item["path"] = str(local_path) if local_path else local_path
    del item["file_name"]
    del item["url"]

    return item


# In[64]:


sample_ds = DatasetDict()

sample_ds["train"] = dataset["train"].select(range(1000)).map(resolve_audio_path, num_proc=n_cpu-1)
sample_ds["test"] = dataset["test"].select(range(200)).map(resolve_audio_path, num_proc=n_cpu)
print(len(sample_ds["train"]), len(sample_ds["test"]))
sample_ds = sample_ds.filter(lambda x: x["path"], num_proc=n_cpu)
print(len(sample_ds["train"]), len(sample_ds["test"]))


# In[65]:


sample_ds["train"][0]


# ## Filter music audio

# In[24]:


audio_classifier = pipeline("audio-classification", model="MarekCech/GenreVim-Music-Detection-DistilHuBERT", device=0)


# In[25]:


def is_music(batch_path):
    audio_arrays = [path["array"] for path in batch_path]
    results = audio_classifier(audio_arrays)  # Assuming `audio_classifier` supports batch input

    # Create a boolean list for filtering: True if 'Music' label > 0.99, False otherwise
    is_music_flags = []
    for result in results:
        is_music = any(label["label"] == "Music" and label["score"] > 0.99 for label in result)
        is_music_flags.append(is_music)

    return is_music_flags

def is_not_music(batch_path):
    audio_arrays = [path["array"] for path in batch_path]
    results = audio_classifier(audio_arrays)  # Assuming `audio_classifier` supports batch input

    # Create a boolean list for filtering: True if 'Music' label > 0.99, False otherwise
    is_not_music_flags = []
    for result in results:
        is_not_music = any(label["label"] == "Non Music" and label["score"] > 0.99 for label in result)
        is_not_music_flags.append(is_not_music)

    return is_not_music_flags


# ### Sample

# In[42]:


sample_ds["train"] = sample_ds["train"].cast_column("path", Audio(sampling_rate=16000))
sample_ds["test"] = sample_ds["test"].cast_column("path", Audio(sampling_rate=16000))


# In[27]:


music_sample_xs = sample_ds.filter(is_music, batched=True, batch_size=64, input_columns="path")
music_sample_xs


# In[28]:


not_music_sample_xs = sample_ds.filter(is_not_music, batched=True, batch_size=64, input_columns="path")
not_music_sample_xs


# In[29]:


def show(item):
    print(item["uni"])
    display(IAudio(item["path"]["array"], rate=16000))


# In[31]:


for item in music_sample_xs["train"]:
    print(item["label"])
    show(item)


# In[32]:


for item in music_sample_xs["test"]:
    print(item["label"])
    show(item)


# In[33]:


sample_ds


# <!-- loaded_ds = DatasetDict.load_from_disk(dataset_path)
# loaded_ds -->

# ### Add Phonetic
# 
# Instead of Wylie (which is used in previous experiments) use simple phonetic

# In[34]:


sample_ds = not_music_sample_xs


# In[36]:


def add_phonetic(item):
    phon = text2phon(item["uni"])
    item["sentence"] = phon
    return item


# In[37]:


sample_ds = sample_ds.map(add_phonetic, num_proc=n_cpu)
sample_ds["train"][0]


# In[38]:


dataset_path = data_path / "tts-training-processed-sample-phono"
sample_ds.save_to_disk(dataset_path)


# In[39]:


loaded_sample_dataset = DatasetDict.load_from_disk(dataset_path)
loaded_sample_dataset["train"][0]


# ## Process All

# In[ ]:


ds = balanced_dataset
ds = concatenate_datasets([ds["train"], ds["test"]])


# In[ ]:


print("#"*200)
print("[INFO] Processing all dataset,", len(ds))


# In[ ]:


# download audio files
ds = ds.map(resolve_audio_path, num_proc=n_cpu)
ds = ds.filter(lambda x: x["path"], num_proc=n_cpu)


# In[ ]:


# load audio
ds = ds.cast_column("path", Audio(sampling_rate=16000))


# In[68]:


# filter out music audio
print("[INFO] Filtering non-music audio...")
ds = ds.filter(is_not_music, batched=True, batch_size=128, input_columns="path")
print("[INFO] Dataset size:", len(ds))


# In[ ]:


ds = get_balance_dataset(ds, total=200000)


# In[ ]:


# add phono
ds = ds.map(add_phonetic, num_proc=n_cpu)


# In[ ]:


# save dataset
dataset_path = data_path / "tts-training-processed-phono"
ds.save_to_disk(dataset_path)


# In[ ]:




