import os
import json

import numpy as np
import torch
import mediapipe as mp


import numpy as np
import torch

from model import Model

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


def get_landmarks1d(landmarks):
    """
    Convert landmarks to 1d tensor, drop visibility and presence

    Args:
        landmarks: list of dict
    Returns:
        landmarks1d: ndarray
    """
    landmarks1d = []
    # flattten landmarks
    for l in landmarks:
        landmarks1d.append(l["x"])
        landmarks1d.append(l["y"])
        landmarks1d.append(l["z"])

    # convert landmarks to tensor
    landmarks1d = np.array(landmarks1d, dtype=np.float32)

    return landmarks1d


def mediapipe_predict(image_file: str, mediapipe_model_path: str):

    # prepare mediapipe settings
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a pose landmarker instance with the video mode:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=mediapipe_model_path),
        running_mode=VisionRunningMode.IMAGE,
        output_segmentation_masks=True,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:

        # Load the input image from an image file.
        mp_image = mp.Image.create_from_file(image_file)
        # Perform pose landmarking on the provided single image.
        # The pose landmarker must be created with the image mode.
        pose_landmarker_result = landmarker.detect(mp_image)

    return pose_landmarker_result


def prepare_mediapiep_input(landmarks_result):

    # save pose landmarks as json
    pose_world_landmarks_json = []

    for lm in landmarks_result.pose_world_landmarks[0]:
        pose_world_landmarks_json.append(
            {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
                "presence": lm.presence,
            }
        )

    mediapiep_input = get_landmarks1d(pose_world_landmarks_json)

    return mediapiep_input


def load_model(saved_model_path: str):

    # load model saved by `torch.save(model.state_dict(), "model.pth")`
    model = Model()

    model.load_state_dict(
        torch.load(saved_model_path, map_location=torch.device("cpu"))
    )

    model.eval()  # set the model to evaluation mode

    return model


def numpy2json(preicted_arr: np.ndarray) -> dict:

    json_result = {bone: {"values": []} for bone in HUMANOID_BONES}

    for n_frame in range(preicted_arr.shape[0]):

        frame_data = preicted_arr[n_frame]

        for i in range(frame_data.shape[0]):

            bone_name = HUMANOID_BONES[i]

            data = frame_data[i].tolist()

            json_result[bone_name]["values"].append(data)

    return json_result


def predict_animation(
    animation_name: str,
    saved_model_path: str,
    mediapipe_model_path: str,
    screenshots_dir: str,
    output_dir: str,
):

    humanoid_name = "dors.glb"
    # animation_name = "Action Idle To Standing Idle.json"

    elevation = 30
    azimuth = 0

    screenshot_path = os.path.join(
        screenshots_dir, humanoid_name, animation_name, str(elevation), str(azimuth)
    )

    image_files = [
        os.path.join(screenshot_path, f) for f in os.listdir(screenshot_path)
    ]

    # sort image_files by {n_frame}.jpg
    image_files.sort(key=lambda x: int(x.split(os.sep)[-1].split(".")[0]))

    model = load_model(saved_model_path)

    mediapiep_inputs = []

    for img in image_files:
        landmarks = mediapipe_predict(img, mediapipe_model_path)

        mediapiep_input = prepare_mediapiep_input(landmarks)
        mediapiep_inputs.append(mediapiep_input)

    # wrap mediapiep_inputs and resnet_inputs as tensor
    mediapiep_inputs = torch.from_numpy(np.array(mediapiep_inputs))

    with torch.no_grad():  # no need to track the gradients
        (
            hip_spine,
            neck_head,
            right_arm,
            right_hand,
            left_arm,
            left_hand,
            right_leg,
            right_foot,
            left_leg,
            left_foot,
        ) = model(mediapiep_inputs)

    preicted = torch.cat(
        (
            hip_spine,
            neck_head,
            right_arm,
            right_hand,
            left_arm,
            left_hand,
            right_leg,
            right_foot,
            left_leg,
            left_foot,
        ),
        1,
    )

    preicted = preicted.reshape(-1, 22, 3)
    preicted = preicted.numpy()

    # print(preicted.shape)

    json_result = numpy2json(preicted)

    # print(json_result)

    # save json to file
    with open(os.path.join(output_dir, f"pred_{animation_name}"), "w") as f:
        json.dump(json_result, f)


if __name__ == "__main__":

    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

    from constants import CHECKPOINT_DIR, HOME_DIR

    predict_animation(
        animation_name="Crouch Turn Left 90.json",
        saved_model_path=os.path.join(CHECKPOINT_DIR, "Model_109.pth"),
        mediapipe_model_path=os.path.join(
            HOME_DIR,
            "Documents",
            "video2motion-dataset",
            "mediapipe",
            "models",
            "pose_landmarker_heavy.task",
        ),
        screenshots_dir=os.path.join(
            HOME_DIR, "Documents", "video2motion", "screenshots"
        ),
        output_dir=os.path.join(
            HOME_DIR,
            "Documents",
            "video2motion-screenshots",
            "anim-player",
            "public",
            "anim-json-euler-pred",
        ),
    )

    pass
