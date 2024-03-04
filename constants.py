import os

HOME_DIR = os.path.expanduser("~")

TRAIN_DATASET_DIR = os.path.join(HOME_DIR, "Documents", "video2motion", "train")

TEST_DATASET_DIR = os.path.join(HOME_DIR, "Documents", "video2motion", "test")

if __name__ == "__main__":

    for f in os.listdir(TRAIN_DATASET_DIR):
        print(f)
        # os.remove(os.path.join(SCREEN_SHOT_DIR, f))
