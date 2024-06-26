DATASETS = ["cedar", "chinese", "dutch", "hindi", "bengali", "gdps", "czech", "all"]
DATASETS_TEST = [
    "cedar_test",
    "chinese_test",
    "dutch_test",
    "hindi_test",
    "bengali_test",
    "gdps_test",
    "czech_test",
    "all_test",
]
FEATURES = [
    "None",
    "strokes",
    "histogram",
    "local",
    "local_solo",
    "wavelet",
    "tri_shape",
    "tri_surface",
    "six_fold",
]
DATASET_NUM_CLASSES = {
    "cedar": 53,
    "hindi": 156,
    "bengali": 97,
    "gdps": 3990,
    "dutch": 60,
    "chinese": 19,
    "all": 0,
    "czech": 34,
    "cedar_test": 1,
    "hindi_test": 4,
    "bengali_test": 3,
    "gdps_test": 10,
    "dutch_test": 4,
    "chinese_test": 1,
    "all_test": 0,
    "czech_test": 1,
}
DATASET_SIGNATURES_PER_PERSON = {
    "cedar_org": 24,
    "cedar_forg": 24,
    "bengali_org": 24,
    "bengali_forg": 30,
    "hindi_org": 24,
    "hindi_forg": 30,
    "gdps_org": 24,
    "gdps_forg": 30,
    "dutch_org": 24,
    "chinese_org": 24,
    "czech_org": 24,
    "czech_forg": 24,
}

MODEL_DIR = "models/server/czech"
