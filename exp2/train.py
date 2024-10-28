from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from pathlib import Path
import multiprocessing
import requests
import sys
import pyewts

from datasets import load_dataset, Audio, Dataset, DatasetDict

from transformers import SpeechT5HifiGan

sanity_check = bool(sys.argv[1]) if len(sys.argv) > 1 else None
if sanity_check:
    print("[INFO] Running in Sanity Checking mode...")
else:
    print("Training preparing...")

converter = pyewts.pyewts()
n_cpu = multiprocessing.cpu_count(); n_cpu
data_path = Path.cwd() / "data"
audio_path = data_path / "audio"
audio_path.mkdir(parents=True, exist_ok=True)

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

dataset_path = data_path / "tts-training-processed"
dataset = DatasetDict.load_from_disk(dataset_path)

if sanity_check:
    dataset["train"] = dataset["train"].select(range(100))
    dataset["test"] = dataset["test"].select(range(20))

dataset = dataset.filter(lambda x: x["path"] and x["path"] != "None", num_proc=n_cpu)
print(dataset["train"][0])

print(f"[INFO] Train({len(dataset['train'])}), Test({len(dataset['test'])})")

dataset['train'] = dataset['train'].cast_column("path", Audio(sampling_rate=16000))
dataset['test'] = dataset['test'].cast_column("path", Audio(sampling_rate=16000))


def to_wylie(example):
    example["sentence"] = converter.toWylie(example["uni"])
    return example

dataset = dataset.map(to_wylie, num_proc=n_cpu)

tokenizer = processor.tokenizer
a = tokenizer.get_vocab().items()

def extract_all_chars(batch):
    all_text = " ".join(batch["sentence"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocabs = dataset['train'].map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=dataset['train'].column_names,
)
vocabs = dataset['test'].map(
    extract_all_chars,
    batched=True,
    batch_size=-1,
    keep_in_memory=True,
    remove_columns=dataset['test'].column_names,
)

dataset_vocab = set(vocabs["vocab"][0])
tokenizer_vocab = {k for k,_ in tokenizer.get_vocab().items()}

replacements = [
    ('_', '_'),
    ('*', 'v'),
    ('`', ';'),
    ('~', ','),
    ('+', ','),
    ('\\', ';'),
    ('|', ';'),
    ('╚',''),
    ('╗',''),
    ('停', ''),
    ('抢', ''),
    ('•', ''), ('0', ''), ('1', ''), ('2', ''), ('3', ''), ('4', ''), ('5', ''), ('6', ''), ('7', ''), ('8', ''), ('9', '')
    
    
]

def cleanup_text(inputs):
    for src, dst in replacements:
        inputs["sentence"] = inputs["sentence"].replace(src, dst)
    return inputs

dataset = dataset.map(cleanup_text, num_proc=n_cpu-1)


import os
import torch
from speechbrain.pretrained import EncoderClassifier

spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

device ="cpu"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name)
)

def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings

def prepare_dataset(batch):
    # load the audio data; if necessary, this resamples the audio to 16kHz
    audio_batch = batch["path"]
    sentence_batch = batch["sentence"]
    audio_array_batch = [audio["array"] for audio in audio_batch]
    sampling_rate = audio_batch[0]["sampling_rate"]

    example = processor(
        text=sentence_batch,
        audio_target=audio_array_batch,
        sampling_rate=sampling_rate,
        return_attention_mask=False,
    )

    example["sentence"] = sentence_batch
    example["labels"] = [label for label in example["labels"]]
    example["path"] = audio_batch
    example["speaker"] = batch['speaker_id']
    example["uni"] = batch["uni"]
    example["speaker_embeddings"] = [create_speaker_embedding(array) for array in audio_array_batch]

    return example

dataset['train'] = dataset['train'].map(
    prepare_dataset, 
    remove_columns=dataset['train'].column_names, 
    batched=True,
    batch_size=32
)

dataset['test'] = dataset['test'].map(
    prepare_dataset, 
    remove_columns=dataset['test'].column_names, 
    batched=True,
    batch_size=32
)


from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:

        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = processor.pad(
            input_ids=input_ids,
            labels=label_features,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor([
                len(feature["input_values"]) for feature in label_features
            ])
            target_lengths = target_lengths.new([
                length - length % model.config.reduction_factor for length in target_lengths
            ])
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch


data_collator = TTSDataCollatorWithPadding(processor=processor)

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./TTS_st5_26102024",  # change to a repo name of your choice
    per_device_train_batch_size=128,
    gradient_accumulation_steps=1,
    num_train_epochs=30,
    learning_rate=1e-5,
    warmup_steps=500,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=16,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=50,
    save_total_limit=20,
    report_to=["wandb"],
    label_names=["labels"],
)


from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    tokenizer=processor.tokenizer,
)
torch.cuda.empty_cache()


trainer.train()

model.push_to_hub('TTS-ST5-26102024')
