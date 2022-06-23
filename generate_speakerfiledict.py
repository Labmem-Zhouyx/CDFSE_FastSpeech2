import os
import json


speakerfile_set = {}
basepath = "preprocessed_data/AISHELL3/"
subsets = ["train.txt", "val.txt", "test.txt"]
for name in subsets:
    with open(os.path.join(basepath, name), "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            basename, speaker, phone_text = line.split("|")[0], line.split("|")[1], line.split("|")[2]
            if speaker in speakerfile_set:
                speakerfile_set[speaker].append("|".join([basename, phone_text]))
            else:
                speakerfile_set[speaker] = ["|".join([basename, phone_text])]

with open(os.path.join(basepath, "speakerfile_dict.json"), "w") as f:
    f.write(json.dumps(speakerfile_set))
