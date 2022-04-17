librispeech_datasets = {
    "train": {
        "clean": ["LibriSpeech/train-clean-100", "LibriSpeech/train-clean-360"],
        "other": ["LibriSpeech/train-other-500"]
    },
    "test": {
        "clean": ["LibriSpeech/test-clean"],
        "other": ["LibriSpeech/test-other"]
    },
    "dev": {
        "clean": ["LibriSpeech/dev-clean"],
        "other": ["LibriSpeech/dev-other"]
    },
}
libritts_datasets = {
    "train": {
        "clean": ["LibriTTS/train-clean-100", "LibriTTS/train-clean-360"],
        "other": ["LibriTTS/train-other-500"]
    },
    "test": {
        "clean": ["LibriTTS/test-clean"],
        "other": ["LibriTTS/test-other"]
    },
    "dev": {
        "clean": ["LibriTTS/dev-clean"],
        "other": ["LibriTTS/dev-other"]
    },
}
###########################################################################################################################################
TIMIT = {
  "TRAIN": ["TIMIT/TRAIN/DR1", "TIMIT/TRAIN/DR2","TIMIT/TRAIN/DR3","TIMIT/TRAIN/DR4",
  "TIMIT/TRAIN/DR5","TIMIT/TRAIN/DR6","TIMIT/TRAIN/DR7","TIMIT/TRAIN/DR8"],
  "TEST" : ["TIMIT/TEST/DR1", "TIMIT/TEST/DR2","TIMIT/TEST/DR3","TIMIT/TEST/DR4",
  "TIMIT/TEST/DR5","TIMIT/TEST/DR6","TIMIT/TEST/DR7","TIMIT/TEST/DR8"]

}
#############################################################################################################################################
voxceleb_datasets = {
    "voxceleb1" : {
        "train": ["VoxCeleb1/wav"],
        "test": ["VoxCeleb1/test_wav"]
    },
    "voxceleb2" : {
        "train": ["VoxCeleb2/dev/aac"],
        "test": ["VoxCeleb2/test_wav"]
    }
}

other_datasets = [
    "LJSpeech-1.1",
    "VCTK-Corpus/wav48",
]

anglophone_nationalites = ["australia", "canada", "ireland", "uk", "usa"]
