from transformers import __version__ as transformers_ver
from tqdm import __version__ as tqdm_ver
from torch import __version__ as torch_ver
from torchaudio import __version__ as torchaudio_ver
from pandas import __version__ as pd_ver
print(f"transformers_ver:\t{transformers_ver}")
print(f"tqdm_ver:\t{tqdm_ver}")
print(f"torch_ver:\t{torch_ver}")
print(f"torchaudio_ver:\t{torchaudio_ver}")
print(f"pandas_ver:\t{pd_ver}")

import os
import torch
import torchaudio
import fire
import pandas as pd
from transformers import AutoModelForCTC, Wav2Vec2Processor

model = AutoModelForCTC.from_pretrained("/app/wav2vec2-xlsr-53-espeak-cv-ft")
processor = Wav2Vec2Processor.from_pretrained("/app/wav2vec2-xlsr-53-espeak-cv-ft")

device_s = f"cuda" if torch.cuda.is_available() else "cpu"
print(f"using device {device_s}")
device = torch.device(device_s)

model = model.to(device)

os.makedirs('./output', exist_ok=True)

def recognizer(fpath):
    try:
        waveform, sample_rate = torchaudio.load(fpath)
        waveform = waveform.to(device)
        logits = model(waveform).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids)[0]
        return pred_str
    except:
        return "SIL"



def _solve_asr(asr_covered_path: str):
    asr_data = pd.read_csv(asr_covered_path)
    audio_dirname = os.path.dirname(asr_covered_path)
    audio_paths = audio_dirname + '/' + asr_data['source'] 
    asr_data["transcription"] = audio_paths.apply(recognizer).str.replace(' ','')
    asr_data["transcription"] = asr_data["transcription"].apply(lambda v: v if v else "SIL")
    asr_data.to_csv("./output/asr-solution.csv", index=False)

def _solve_translation(translation_covered_path: str):
    translation_data = pd.read_csv(translation_covered_path)
    translation_data["translation"] = translation_data["source"]
    translation_data["translation"] = translation_data["translation"].apply(lambda v: v if v else "UNK")
    translation_data.to_csv("./output/translation-solution.csv", index=False)


def solve_tasks(asr_path: str, translation_path: str = None):
    _solve_asr(asr_path)
    if translation_path is not None:
        _solve_translation(translation_path)

if __name__ == '__main__':
    fire.Fire()
