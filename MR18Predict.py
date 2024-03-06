import os
import json

import numpy as np
import torch
import mediapipe as mp


import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from constants import CHECKPOINT_DIR, HOME_DIR, SCREENSHOT_DIR
from MR18Model import MR18Model


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


def mediapipe_predict(image_file):

    model_path = os.path.join(
        HOME_DIR,
        "Repos",
        "video2motion-dataset",
        "mediapipe",
        "models",
        "pose_landmarker_heavy.task",
    )

    # prepare mediapipe settings
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Create a pose landmarker instance with the video mode:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
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


def load_model():
    saved_model_path = os.path.join(CHECKPOINT_DIR, "MR18Model_24.pth")

    # load model saved by `torch.save(model.state_dict(), "model.pth")`
    model = MR18Model()

    model.load_state_dict(
        torch.load(saved_model_path, map_location=torch.device("cpu"))
    )

    model.eval()  # set the model to evaluation mode

    return model


def extract_feature_vector(
    img_filename: str,
):

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    input_image = Image.open(img_filename)

    input_tensor = preprocess(input_image)
    # create a mini-batch as expected by the model
    input_batch = input_tensor.unsqueeze(0)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    t_img = input_batch

    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512

    my_embedding = torch.zeros(1, 512, 1, 1)

    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)

    # Use the model object to select the desired layer
    layer = model._modules.get("avgpool")

    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)

    # 6. Run the model on our transformed image
    model(t_img)

    # 7. Detach our copy function from the layer
    h.remove()

    # 8. Return the feature vector
    return my_embedding.squeeze().numpy()


def predict_animation(animation_name):

    humanoid_name = "dors.glb"
    # animation_name = "Action Idle To Standing Idle.json"

    elevation = 30
    azimuth = 0

    screenshot_path = os.path.join(
        SCREENSHOT_DIR, humanoid_name, animation_name, str(elevation), str(azimuth)
    )

    image_files = [
        os.path.join(screenshot_path, f) for f in os.listdir(screenshot_path)
    ]

    # sort image_files by {n_frame}.jpg
    image_files.sort(key=lambda x: int(x.split(os.sep)[-1].split(".")[0]))

    model = load_model()

    mediapiep_inputs = []
    resnet_inputs = []

    for img in image_files:
        landmarks = mediapipe_predict(img)

        mediapiep_input = prepare_mediapiep_input(landmarks)
        mediapiep_inputs.append(mediapiep_input)

        resnet_input = extract_feature_vector(img)
        resnet_inputs.append(resnet_input)

    # wrap mediapiep_inputs and resnet_inputs as tensor
    mediapiep_inputs = torch.from_numpy(np.array(mediapiep_inputs))
    resnet_inputs = torch.from_numpy(np.array(resnet_inputs))

    with torch.no_grad():  # no need to track the gradients

        output = model(mediapiep_inputs, resnet_inputs)

    # save it to local file
    np.save("output.npy", output.numpy())


def output2anim_json(output_file):

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

    json_result = {bone: {"values": []} for bone in HUMANOID_BONES}

    # print(json_result)

    output = np.load(output_file)

    # print(output)
    # return

    for n_frame in range(output.shape[0]):

        frame_data = output[n_frame]

        for i in range(frame_data.shape[0]):

            bone_name = HUMANOID_BONES[i]

            data = frame_data[i].tolist()

            json_result[bone_name]["values"].append(data)

    # print(json_result)
    # save json to file
    with open("output.json", "w") as f:
        json.dump(json_result, f)


if __name__ == "__main__":

    output2anim_json("output.npy")

    pass
