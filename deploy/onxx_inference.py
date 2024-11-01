from pathlib import Path

import onnxruntime as ort
import numpy as np
import soundfile as sf
from transformers import SpeechT5Processor

import time


start_time = time.time()


encoder_path = "/home/jovyan/speecht5-tts-01/onnx/encoder_model.onnx"
decoder_path = "/home/jovyan/speecht5-tts-01/onnx/decoder_model_merged.onnx"
postnet_and_vocoder_path = "/home/jovyan/speecht5-tts-01/onnx/decoder_postnet_and_vocoder.onnx"
speaker_embeddings_path = "/home/jovyan/speech_t5_tts/deploy/female_2.npy"

encoder = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
decoder = ort.InferenceSession(decoder_path, providers=["CPUExecutionProvider"])
postnet_and_vocoder = ort.InferenceSession(postnet_and_vocoder_path, providers=["CPUExecutionProvider"])

def add_fake_pkv(inputs):
    shape = (1, 12, 0, 64)
    for i in range(6):
        inputs[f"past_key_values.{i}.encoder.key"] = np.zeros(shape).astype(np.float32)
        inputs[f"past_key_values.{i}.encoder.value"] = np.zeros(shape).astype(np.float32)
        inputs[f"past_key_values.{i}.decoder.key"] = np.zeros(shape).astype(np.float32)
        inputs[f"past_key_values.{i}.decoder.value"] = np.zeros(shape).astype(np.float32)
    return inputs

def add_real_pkv(inputs, previous_outputs, cross_attention_pkv):
    for i in range(6):
        inputs[f"past_key_values.{i}.encoder.key"] = cross_attention_pkv[f"present.{i}.encoder.key"]
        inputs[f"past_key_values.{i}.encoder.value"] = cross_attention_pkv[f"present.{i}.encoder.value"]
        inputs[f"past_key_values.{i}.decoder.key"] = previous_outputs[f"present.{i}.decoder.key"]
        inputs[f"past_key_values.{i}.decoder.value"] = previous_outputs[f"present.{i}.decoder.value"]
    return inputs

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")

text = "stobs chen rgyal khab nyi shu'i lhan tshogs thog la rgya nag gzhung gis bod nang rig gzhung rtsa gtor kyi srid byus khag dpar ris thog nas las 'gul spel ba'i skor__'jam dbyangs rgya mtsho lags kyis snyan sgron gnang gi red/"

inputs = processor(text=text, return_tensors="np")

inp = {
    "input_ids": inputs["input_ids"]
}

outputs = encoder.run(None, inp)
outputs = {output_key.name: outputs[idx] for idx, output_key in enumerate(encoder.get_outputs())}

encoder_last_hidden_state = outputs["encoder_outputs"]
encoder_attention_mask = outputs["encoder_attention_mask"]

minlenratio = 0.0
maxlenratio = 20.0
reduction_factor = 2
threshold = 0.9
num_mel_bins = 80

maxlen = int(encoder_last_hidden_state.shape[1] * maxlenratio / reduction_factor)
minlen = int(encoder_last_hidden_state.shape[1] * minlenratio / reduction_factor)

spectrogram = []
cross_attentions = []
past_key_values = None
idx = 0
cross_attention_pkv = None
use_cache_branch = False

speaker_embeddings = np.load(speaker_embeddings_path).astype(np.float32)

while True:
    idx += 1

    decoder_inputs = {}
    decoder_inputs["use_cache_branch"] = np.array([use_cache_branch])
    decoder_inputs["encoder_attention_mask"] = encoder_attention_mask
    decoder_inputs["speaker_embeddings"] = speaker_embeddings

    if not use_cache_branch:
        decoder_inputs = add_fake_pkv(decoder_inputs)
        decoder_inputs["output_sequence"] = np.zeros((1, 1, num_mel_bins)).astype(np.float32)
        use_cache_branch = True
        decoder_inputs["encoder_hidden_states"] = encoder_last_hidden_state
    else:
        decoder_inputs = add_real_pkv(decoder_inputs, decoder_outputs, cross_attention_pkv)
        decoder_inputs["output_sequence"] = decoder_outputs["output_sequence_out"]
        decoder_inputs["encoder_hidden_states"] = np.zeros((1, 0, 768)).astype(np.float32)  # useless when cross-attention KV has already been computed

    decoder_outputs = decoder.run(None, decoder_inputs)
    decoder_outputs = {output_key.name: decoder_outputs[idx] for idx, output_key in enumerate(decoder.get_outputs())}

    if idx == 1:  # i.e. use_cache_branch = False
        cross_attention_pkv = {key: val for key, val in decoder_outputs.items() if ("encoder" in key and "present" in key)}

    prob = decoder_outputs["prob"]
    spectrum = decoder_outputs["spectrum"]

    spectrogram.append(spectrum)

    # print("prob", prob)

    # Finished when stop token or maximum length is reached.
    if idx >= minlen and (int(sum(prob >= threshold)) > 0 or idx >= maxlen):
        # print("len spectrogram", len(spectrogram))
        spectrogram = np.concatenate(spectrogram)
        vocoder_output = postnet_and_vocoder.run(None, {"spectrogram": spectrogram})
        break

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")

sf.write("speech.wav", vocoder_output[0], samplerate=16000)