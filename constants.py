import os

HOME_DIR = os.path.expanduser("~")

TRAIN_DATASET_DIR = os.path.join(HOME_DIR, "Documents", "video2motion", "train")

TEST_DATASET_DIR = os.path.join(HOME_DIR, "Documents", "video2motion", "test")


MEDIAPIPE_QUEUE_DATA_DIR = os.path.join(
    HOME_DIR, "Documents", "video2motion", "queue_data", "mediapipe"
)

RESNET_QUEUE_DATA_DIR = os.path.join(
    HOME_DIR, "Documents", "video2motion", "queue_data", "resnet"
)

ANIM_EULER_QUEUE_DATA_DIR = os.path.join(
    HOME_DIR, "Documents", "video2motion", "queue_data", "anim_euler"
)

MEDIAPIPE_JOINED_DIR = os.path.join(
    HOME_DIR, "Documents", "video2motion", "joined", "mediapipe"
)

RESNET_JOINED_DIR = os.path.join(
    HOME_DIR, "Documents", "video2motion", "joined", "resnet"
)

ANIM_EULER_JOINED_DIR = os.path.join(
    HOME_DIR, "Documents", "video2motion", "joined", "anim_euler"
)

if __name__ == "__main__":

    for f in os.listdir(TRAIN_DATASET_DIR):
        print(f)
        # os.remove(os.path.join(SCREEN_SHOT_DIR, f))
