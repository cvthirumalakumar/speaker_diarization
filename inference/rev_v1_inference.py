from pyannote.audio import Pipeline
import torch
import glob
from tqdm import tqdm
import os 

pipeline = Pipeline.from_pretrained(
    "Revai/reverb-diarization-v1",
    use_auth_token="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")

pipeline.to(torch.device("cuda"))

# diarization = pipeline("/mnt/sd1/kumar/code/speaker_diarisation/primock57/output/mixed_audio/day1_consultation01.wav")

audio_folder_path = "/mnt/sd1/kumar/code/speaker_diarisation/primock57/output/mixed_audio"
output_path = "rev_v1_preds"
os.makedirs(output_path, exist_ok=True)
for audio in tqdm(glob.glob(audio_folder_path+"/*.wav")):
    file_name = audio.split("/")[-1].split(".")[0]
    diarization = pipeline(audio)
    with open(output_path+"/"+file_name+".rttm", "w") as f:
        diarization.write_rttm(f)