import os
import re
import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm

from text import _clean_text


def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    speaker = "DataBaker"
    text_withprosody = ""
    with open(os.path.join(in_dir, "ProsodyLabeling", "000001-010000.txt"), encoding="utf-8") as f:
        content = f.readlines()
        num = int(len(content) // 2)
        for idx in tqdm(range(num)):
            base_name, rawtext, text = _parse_cn_prosody_label(content[idx * 2], content[idx * 2 + 1])

            wav_path = os.path.join(in_dir, "Wave", "{}.wav".format(base_name))
            if os.path.exists(wav_path):
                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wav, _ = librosa.load(wav_path, sampling_rate)
                wav = wav / max(abs(wav)) * max_wav_value
                wavfile.write(
                    os.path.join(out_dir, speaker, "{}.wav".format(base_name)),
                    sampling_rate,
                    wav.astype(np.int16),
                )
                with open(
                    os.path.join(out_dir, speaker, "{}.lab".format(base_name)),
                    "w",
                ) as f1:
                    f1.write(text)


def _parse_cn_prosody_label(text, pinyin, use_prosody=True):
    """
    Parse label from text and pronunciation lines with prosodic structure labelings

    Input text:    100001 妈妈#1当时#1表示#3，儿子#1开心得#2像花儿#1一样#4。
    Input pinyin:  ma1 ma1 dang1 shi2 biao3 shi4 er2 zi5 kai1 xin1 de5 xiang4 huar1 yi2 yang4
    Return sen_id: 100001
    Return pinyin: ma1 ma1 #1 dang1 shi2 #1 biao3 shi4 #3 er2 zi5 #1 kai1 xin1 de5 #2 xiang4 huar1 #1 yi2 yang4 #4
    Args:
        - text: Chinese characters with prosodic structure labeling, begin with sentence id for wav and interval file
        - pinyin: Pinyin pronunciations, with tone 1-5
        - use_prosody: Whether the prosodic structure labeling information will be used
    Returns:
        - (sen_id, pinyin&tag): latter contains pinyin string with optional prosodic structure tags
    """

    text = text.strip()
    pinyin = pinyin.strip()
    if len(text) == 0:
        return None

    # remove punctuations
    text = re.sub('[“”、，。：；？！—…#（）]', '', text)

    # split into sub-terms
    sen_id, texts = text.split()
    phones = pinyin.split()

    # prosody boundary tag (SYL: 音节, PWD: 韵律词, PPH: 韵律短语, IPH: 语调短语, SEN: 语句)
    if use_prosody:
        SYL = '-'
        PWD = ' '
        PPH = '/'
        IPH = ','
        SEN = '.'
    else:
        SYL = PWD = PPH = IPH = SEN = ' '

    # parse details
    pinyin = ''
    text_aft = ''
    i = 0  # texts index
    j = 0  # phones index
    b = 1  # left is boundary
    while i < len(texts):
        if texts[i].isdigit():
            if texts[i] == '1':
                pinyin += PWD  # Prosodic Word, 韵律词边界
            if texts[i] == '2':
                pinyin += PPH  # Prosodic Phrase, 韵律短语边界
            if texts[i] == '3':
                pinyin += IPH  # Intonation Phrase, 语调短语边界
                text_aft += IPH
            if texts[i] == '4':
                pinyin += SEN  # Sentence, 语句结束
                text_aft += SEN
            b = 1
            i += 1

        elif texts[i] != '儿' or j == 0 or not _is_erhua(phones[j - 1][:-1]):  # Chinese segment
            if b == 0: pinyin += SYL  # Syllable, 音节边界（韵律词内部）
            pinyin += phones[j]
            text_aft += texts[i]
            b = 0
            i += 1
            j += 1

        else:  # 儿化音
            text_aft += texts[i]
            i += 1

    return (sen_id, text_aft, pinyin)


def _is_erhua(pinyin):
	"""
	Decide whether pinyin (without tone number) is retroflex (Erhua)
	"""
	if len(pinyin)<=1 or pinyin == 'er':
		return False
	elif pinyin[-1] == 'r':
		return True
	else:
		return False