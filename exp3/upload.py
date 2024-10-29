from huggingface_hub import HfApi

api = HfApi()

api.upload_large_folder(
    repo_id="openpecha/tts-training-processed-phono",
    repo_type="dataset",
    folder_path="/data/volume/tts-training-processed-phono",
)