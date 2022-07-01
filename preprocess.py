import argparse

import yaml

from preprocessor import preprocessor
from preprocessor import preprocessor_multiset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    if "LibriTTS" in config["dataset"]:
        preprocessor = preprocessor_multiset.Preprocessor(config)
    else:
        preprocessor = preprocessor.Preprocessor(config)
    preprocessor.build_from_path()
