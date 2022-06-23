[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FLabmem-Zhouyx%2FCDFSE_FastSpeech2&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)


#  CDFSE_FastSpeech2
This repo contains code accompanying the paper "Content-Dependent Fine-Grained Speaker Embedding for Zero-Shot Speaker Adaptation in Text-to-Speech Synthesis", which is implemented based on [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (much thanks!).

*2022-06-15 Update: This work has been accepted by Interspeech 2022.* 

## [Samples](https://thuhcsi.github.io/interspeech2022-cdfse-tts/) | [Paper](https://arxiv.org/abs/2204.00990)

## Usage

### 0. Dataset
 1. Mandarin: [AISHELL3](https://www.openslr.org/93/)
 2. English: [LibriTTS](http://www.openslr.org/60/) 

### 1. Environment setup
```bash
pip3 install -r requirements.txt
```

### 2. Data pre-processing
Please refer to [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)  for more details.

For example, first run
```bash
python3 prepare_align.py config/AISHELL3/preprocess.yaml
```
Then download TextGrid Files or use MFA to align the corpus, and put TextGrid Files in your [PREPROCESSED_DATA_PATH] like preprocessed_data/AISHELL3/TextGrid/. 

Finally, run the preprocessing script 
```bash
python3 preprocess.py config/AISHELL3/preprocess.yaml
```
*In addition:*
 1. we have split the train, val, and test sets in preprocessed_data/[DATASET]/*. So you can put them directly in your [PREPROCESSED_DATA_PATH] after data-preprocessing, or re-split them yourself. 
 2. We have provided "speakerfile_dict.json" in preprocessed_data/[DATASET]/* (used in dataset.py for randomly loading reference speech), and you can generate it with generate_speakerfiledict.py.
 3. We have provided some hifigan pretrained parameters in hifigan/pretrained/*, you can just load them (remember to unzip the *.zip file) or use your own well-trained vocoder in utils/model.py.

### 3. Training
Train the model
```bash
python3 train.py -p config/AISHELL3/preprocess.yaml -m config/AISHELL3/model.yaml -t config/AISHELL3/train.yaml 
```
*If you find the PhnCls Loss doesn't seem to be trending down or is not noticeable, try manually adjusting the symbol dicts in text/symbols.py (only contains relevant phonemes) to make phoneme classification work better, and this may solve the problem.*

(Optional) Use tensorboard
```bash
tensorboard --logdir output/log/AISHELL3
```

### 4. Inference
For batch:
```bash
python3 synthesize.py --source synbatch_chinese.txt --restore_step 250000 --mode batch -p config/AISHELL3/preprocess.yaml -m config/AISHELL3/model.yaml -t config/AISHELL3/train.yaml 
```
	
For single:
```bash
# For Mandarin
python3 synthesize.py --text "清华大学人机语音交互实验室，聚焦人工智能场景下的智能语音交互技术研究。" --ref [REF_SPEECH_PATH.wav] --restore_step 250000 --mode single -p config/AISHELL3/preprocess.yaml -m config/AISHELL3/model.yaml -t config/AISHELL3/train.yaml 
# For English
python3 synthesize.py --text "Human Computer Speech Interaction Lab at Tsinghua University, targets artificial intelligence technologies for smart voice user interface." --ref [REF_SPEECH_PATH.wav] --restore_step 250000 --mode single -p config/LibriTTS/preprocess.yaml -m config/LibriTTS/model.yaml -t config/LibriTTS/train.yaml 
```


## Implementation changes
- (2022-06-20) Instance Normalization is adopted in Mel Content Encoder for better performance.
- (2022-06-01) Support English setting: LibriTTS multi-speaker dataset (train-clean-100 + dev-clean + test-clean).  
- (2022-04-27) Support directly using wavfile (*.wav) as reference speech instead of mel-spectrogram numpy file in single mode. 

## References

- [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)
- [jik876/HiFi-GAN](https://github.com/jik876/hifi-gan)
	
## Citation
```
@misc{zhou2022content,
  title={Content-Dependent Fine-Grained Speaker Embedding for Zero-Shot Speaker Adaptation in Text-to-Speech Synthesis}, 
  author={Zhou, Yixuan and Song, Changhe and Li, Xiang and Zhang, Luwen and Wu, Zhiyong and Bian, Yanyao and Su, Dan and Meng, Helen},
  year={2022},
  eprint={2204.00990},
  archivePrefix={arXiv},
  primaryClass={eess.AS}
}
```
