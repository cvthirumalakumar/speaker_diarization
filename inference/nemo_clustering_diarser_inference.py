import json
import glob
from tqdm import tqdm
import os 
from omegaconf import OmegaConf
from nemo.collections.asr.models import ClusteringDiarizer
import wget


print("Creating manifest file")
audio_folder_path = "/mnt/sd1/kumar/code/speaker_diarisation/primock57/output/mixed_audio"
with open('primock57_manifest.json','w') as fp:
    for audio in glob.glob(audio_folder_path+"/*.wav"):
        meta = {
                'audio_filepath': audio,
                'offset': 0,
                'duration':None,
                'label': 'infer',
                'text': '-',
                'num_speakers': 2,
                'rttm_filepath': None,
                'uem_filepath' : None
                }
        json.dump(meta,fp)
        fp.write('\n')
        
clustering_output_dir = "nemo_clustering_preds"
# os.makedirs(clustering_output_dir, exist_ok=True)
data_dir = 'data'
MODEL_CONFIG = os.path.join(data_dir,'diar_infer_telephonic.yaml')
if not os.path.exists(MODEL_CONFIG):
    config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
    MODEL_CONFIG = wget.download(config_url,data_dir)
config = OmegaConf.load(MODEL_CONFIG)
# print(OmegaConf.to_yaml(config))
pretrained_vad = 'vad_multilingual_marblenet'
pretrained_speaker_model = 'titanet_large'

config.diarizer.manifest_filepath = 'primock57_manifest.json'
config.diarizer.out_dir = clustering_output_dir # Directory to store intermediate files and prediction outputs
config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5,1.25,1.0,0.75,0.5]
config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75,0.625,0.5,0.375,0.1]
config.diarizer.speaker_embeddings.parameters.multiscale_weights= [1,1,1,1,1]
config.num_workers = 1 # Workaround for multiprocessing hanging with ipython issue
config.diarizer.oracle_vad = False # compute VAD provided with model_path to vad config
config.diarizer.clustering.parameters.oracle_num_speakers=False
config.diarizer.vad.model_path = pretrained_vad
config.diarizer.vad.parameters.onset = 0.8
config.diarizer.vad.parameters.offset = 0.6
config.diarizer.vad.parameters.pad_offset = -0.05

print("Diarizing")
sd_model = ClusteringDiarizer(cfg=config)
sd_model.diarize()