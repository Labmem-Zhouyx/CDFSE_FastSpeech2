import argparse
import os
import yaml

from preprocessor import ljspeech, aishell3, libritts, databaker


def main(config):
    if "AISHELL3" in config["dataset"]:
        aishell3.prepare_align(config)
    if "LibriTTS" in config["dataset"]:
        corpus_path = config["path"]["corpus_path"]
        raw_path = config["path"]["raw_path"]
        for dmode, dset in config["subsets"].items():
            config["path"]["corpus_path"] = os.path.join(corpus_path, dset)
            config["path"]["raw_path"] = os.path.join(raw_path, dset)
            libritts.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)

