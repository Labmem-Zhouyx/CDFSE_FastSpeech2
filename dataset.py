import json
import math
import os
import re
import random

import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last
        self.shuffle_refmel = train_config["dataset"]["shuffle_refmel"]
        self.sample_refmel = train_config["dataset"]["sample_refmel"]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)
        ref_mel, ref_linguistic = self.get_reference(mel, phone, duration)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "ref_mel": ref_mel,
            "ref_linguistic": ref_linguistic,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }

        return sample

    def get_reference(self, mel, phone, duration):
        mel_slices = []
        linguistic_slices = []
        start = 0
        for i in range(phone.shape[0]):
            linguistic_slices.append([phone[i]] * duration[i])
            mel_slices.append(mel[start: start + duration[i]])
            start += duration[i]

        # random shuffle by phoneme duration
        if self.shuffle_refmel:
            index = list(range(duration.shape[0]))
            if self.sample_refmel:  # introduce random length to ref mel
                tmp = []
                for i in index:
                    tmp += [i] * random.randint(0, 2)
                if len(tmp) > 0:
                    index = tmp
            random.shuffle(index)
            mel_slices_shuffle = []
            linguistic_slices_shuffle = []
            for i in index:
                mel_slices_shuffle.append(mel_slices[i])
                linguistic_slices_shuffle.append(linguistic_slices[i])
            mel_post = np.concatenate(mel_slices_shuffle)
            linguistic = np.concatenate(linguistic_slices_shuffle)
        else:
            mel_post = np.concatenate(mel_slices)
            linguistic = np.concatenate(linguistic_slices)

        return mel_post, linguistic

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]
        ref_mels = [data[idx]["ref_mel"] for idx in idxs]
        ref_linguistics = [data[idx]["ref_linguistic"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])
        ref_mel_lens = np.array([ref_mel.shape[0] for ref_mel in ref_mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        ref_mels = pad_2D(ref_mels)
        ref_linguistics = pad_1D(ref_linguistics)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)

        return (
            ids,
            raw_texts,          
            texts,
            text_lens,
            max(text_lens),
            ref_mels,
            ref_mel_lens,
            mels,
            mel_lens,
            max(mel_lens),
            pitches,
            energies,
            durations,
            ref_linguistics,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.basename, self.raw_text, self.ref_mel = self.process_meta(
            filepath
        )
        self.lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    def __len__(self):
        return len(self.raw_text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        raw_text = self.raw_text[idx]
        ref_mel_path = self.ref_mel[idx]
        ref_mel = np.load(os.path.join(self.preprocessed_path, "mel", ref_mel_path))
        phone = self.preprocess_pinyin(raw_text)

        return (basename, phone, raw_text, ref_mel)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            raw_text = []
            ref_mel = []
            for line in f.readlines():
                n, r, m = line.strip("\n").split("|")
                name.append(n)
                raw_text.append(r)
                ref_mel.append(m)
            return name, raw_text, ref_mel

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        texts = [d[1] for d in data]
        raw_texts = [d[2] for d in data]
        ref_mels = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])
        ref_mel_lens = np.array([ref_mel.shape[0] for ref_mel in ref_mels])

        texts = pad_1D(texts)
        ref_mels = pad_2D(ref_mels)
        
        return ids, raw_texts, texts, text_lens, max(text_lens), ref_mels, ref_mel_lens

    def preprocess_pinyin(self, text):
        
        phones = []
        pinyins = text.split(' ')
        for p in pinyins:
            if p in self.lexicon:
                phones += self.lexicon[p]
            else:
                phones.append("sp")

        phones = "{" + " ".join(phones) + "}"
        sequence = np.array(
            text_to_sequence(phones, [])
        )

        return np.array(sequence)

        
if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.utils import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
