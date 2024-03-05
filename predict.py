import os
import json

import numpy as np
import torch


from MediapipeResnet18JoinedModel import MediapipeResnet18JoinedModel

if __name__ == "__main__":

    saved_model_path = os.path.join("models", "model_24.pth")

    # load model saved by `torch.save(model.state_dict(), "model.pth")`
    model = MediapipeResnet18JoinedModel()

    model.load_state_dict(
        torch.load(saved_model_path, map_location=torch.device("cpu"))
    )

    model.eval()  # set the model to evaluation mode

    humanoid_name = "dors.glb"
    animation_name = "Action Idle To Standing Idle.json"

    elevation = 30
    azimuth = 0

    mediapipe_path = (
        f"mediapipe/{humanoid_name}/{animation_name}/{elevation}/{azimuth}/"
    )

    print(model)

    # # ger all mediapipe prediected result from oss for a given animation

    # world_landmarks_json = {}

    # # list all files in the oss folder
    # for obj in oss2.ObjectIterator(bucket, prefix=mediapipe_path):
    #     # if obj.key.endswith("world_landmarks.json"):
    #     if obj.key.endswith("world_landmarks.json"):

    #         # use regexp to get the n_frame
    #         n_frame = int(obj.key.split("/")[-2])

    #         world_landmarks_json[n_frame] = obj.key
    #         # world_landmarks = json.loads(obj.read())
    #         # landmarks1d = get_landmarks1d(world_landmarks)

    #         # # predict inputs[0] using the model
    #         # with torch.no_grad():  # no need to track the gradients
    #         #     prediction = model(landmarks1d)

    #     # print(obj.key)

    # features = []

    # # sort world_landmarks_json
    # for _, v in sorted(world_landmarks_json.items()):

    #     # todo load the world_landmarks_json from oss
    #     landmarks_obj = bucket.get_object(v)

    #     world_landmarks = json.loads(landmarks_obj.read())

    # # # predict inputs[0] using the model
    # # with torch.no_grad():  # no need to track the gradients
    # #     prediction = model(inputs)
