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

CHECKPOINT_DIR = os.path.join(
    HOME_DIR, "Documents", "video2motion-train", "checkpoints"
)

SCREENSHOT_DIR = os.path.join(HOME_DIR, "Documents", "video2motion", "screenshots")

# BASE_DIR = os.path.join(os.path.expanduser("~"), "Documents")
BASE_DIR = "D:\\"

BLAZEPOSE_KEYPOINTS = {
    0: "NOSE",
    1: "LEFT_EYE_INNER",
    2: "LEFT_EYE",
    3: "LEFT_EYE_OUTER",
    4: "RIGHT_EYE_INNER",
    5: "RIGHT_EYE",
    6: "RIGHT_EYE_OUTER",
    7: "LEFT_EAR",
    8: "RIGHT_EAR",
    9: "LEFT_RIGHT",
    10: "RIGHT_LEFT",
    11: "LEFT_SHOULDER",
    12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW",
    14: "RIGHT_ELBOW",
    15: "LEFT_WRIST",
    16: "RIGHT_WRIST",
    17: "LEFT_PINKY",
    18: "RIGHT_PINKY",
    19: "LEFT_INDEX",
    20: "RIGHT_INDEX",
    21: "LEFT_THUMB",
    22: "RIGHT_THUMB",
    23: "LEFT_HIP",
    24: "RIGHT_HIP",
    25: "LEFT_KNEE",
    26: "RIGHT_KNEE",
    27: "LEFT_ANKLE",
    28: "RIGHT_ANKLE",
    29: "LEFT_HEEL",
    30: "RIGHT_HEEL",
    31: "LEFT_FOOT_INDEX",
    32: "RIGHT_FOOT_INDEX",
}


HUMANOID_BONES = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
]

if __name__ == "__main__":

    for f in os.listdir(TRAIN_DATASET_DIR):
        print(f)
        # os.remove(os.path.join(SCREEN_SHOT_DIR, f))
