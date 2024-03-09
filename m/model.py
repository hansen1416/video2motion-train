import torch
from torch import nn

"""
input joints indices from mediapipe

0: "NOSE",1: "LEFT_EYE_INNER",2: "LEFT_EYE",3: "LEFT_EYE_OUTER",4: "RIGHT_EYE_INNER",
5: "RIGHT_EYE",6: "RIGHT_EYE_OUTER",7: "LEFT_EAR",8: "RIGHT_EAR", 9: "LEFT_RIGHT",10: "RIGHT_LEFT",
11: "LEFT_SHOULDER",12: "RIGHT_SHOULDER",
13: "LEFT_ELBOW",14: "RIGHT_ELBOW",
15: "LEFT_WRIST",16: "RIGHT_WRIST",
17: "LEFT_PINKY",18: "RIGHT_PINKY",
19: "LEFT_INDEX",20: "RIGHT_INDEX",
21: "LEFT_THUMB",22: "RIGHT_THUMB",
23: "LEFT_HIP",24: "RIGHT_HIP",
25: "LEFT_KNEE",26: "RIGHT_KNEE",
27: "LEFT_ANKLE",28: "RIGHT_ANKLE",
29: "LEFT_HEEL",30: "RIGHT_HEEL",
31: "LEFT_FOOT_INDEX",32: "RIGHT_FOOT_INDEX",

target joints indices from anim_euler

0: "Hips", 1: "Spine", 2: "Spine1", 3: "Spine2", 4: "Neck", 5: "Head",
6: "RightShoulder", 7: "RightArm", 8: "RightForeArm", 9: "RightHand",
10: "LeftShoulder", 11: "LeftArm", 12: "LeftForeArm", 13: "LeftHand",
14: "RightUpLeg", 15: "RightLeg", 16: "RightFoot", 17: "RightToeBase",
18: "LeftUpLeg", 19: "LeftLeg", 20: "LeftFoot", 21: "LeftToeBase",


[11, 12, 23, 24] => [0, 1, 2, 3]
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] => [4, 5]
[11, 12, 14, 16, 18, 20, 22] => [6, 7, 8, 9]
[11, 12, 13, 15, 17, 19, 21] => [10, 11, 12, 13]
[23, 24, 26, 28, 30, 32] => [14, 15, 16, 17]
[23, 24, 25, 27, 29, 31] => [18, 19, 20, 21]
"""


class Model(nn.Module):
    def __init__(self):

        super(Model, self).__init__()
        # net1 include joints 11: "LEFT_SHOULDER",12: "RIGHT_SHOULDER", 23: "LEFT_HIP",24: "RIGHT_HIP"
        # to predict 0: "Hips", 1: "Spine", 2: "Spine1", 3: "Spine2",
        self.net1 = nn.Sequential(
            nn.Linear(12, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 12),
        )
        # net2 include joints 1: "LEFT_EYE_INNER",2: "LEFT_EYE",3: "LEFT_EYE_OUTER",4: "RIGHT_EYE_INNER",
        # 5: "RIGHT_EYE",6: "RIGHT_EYE_OUTER",7: "LEFT_EAR",8: "RIGHT_EAR", 9: "LEFT_RIGHT",
        # 10: "RIGHT_LEFT",11: "LEFT_SHOULDER",12: "RIGHT_SHOULDER"
        # to predict 4: "Neck", 5: "Head",
        self.net2 = nn.Sequential(
            nn.Linear(36, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 6),
        )

        # net3 include joints 11: "LEFT_SHOULDER",12: "RIGHT_SHOULDER",14: "RIGHT_ELBOW",
        # 16: "RIGHT_WRIST",18: "RIGHT_PINKY",20: "RIGHT_INDEX",22: "RIGHT_THUMB"
        # to predict 6: "RightShoulder", 7: "RightArm", 8: "RightForeArm", 9: "RightHand",
        self.net3 = nn.Sequential(
            nn.Linear(21, 36),
            nn.Tanh(),
            nn.Linear(36, 16),
            nn.Tanh(),
            nn.Linear(16, 12),
        )

        # net4 include joints 11: "LEFT_SHOULDER",12: "RIGHT_SHOULDER",13: "LEFT_ELBOW",
        # 15: "LEFT_WRIST",17: "LEFT_PINKY",19: "LEFT_INDEX",21: "LEFT_THUMB"
        # to predict 10: "LeftShoulder", 11: "LeftArm", 12: "LeftForeArm", 13: "LeftHand",
        self.net4 = nn.Sequential(
            nn.Linear(21, 36),
            nn.Tanh(),
            nn.Linear(36, 16),
            nn.Tanh(),
            nn.Linear(16, 12),
        )

        # net6 include joints 23: "LEFT_HIP",24: "RIGHT_HIP",26: "RIGHT_KNEE",28: "RIGHT_ANKLE",
        # 30: "RIGHT_HEEL",32: "RIGHT_FOOT_INDEX"
        # to predict 14: "RightUpLeg", 15: "RightLeg", 16: "RightFoot", 17: "RightToeBase",
        self.net5 = nn.Sequential(
            nn.Linear(18, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 12),
        )

        # net5 include joints 23: "LEFT_HIP",24: "RIGHT_HIP",25: "LEFT_KNEE",27: "LEFT_ANKLE",
        # 29: "LEFT_HEEL",31: "LEFT_FOOT_INDEX"
        # to predict 18: "LeftUpLeg", 19: "LeftLeg", 20: "LeftFoot", 21: "LeftToeBase",
        self.net6 = nn.Sequential(
            nn.Linear(18, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 12),
        )

    def forward(self, x):

        # Extract the required joints
        x1 = x.view(-1, 33, 3)[:, [11, 12, 23, 24]]

        # pass x1 to self.net1
        x1 = self.net1(x1.view(-1, 12))

        x2 = x.view(-1, 33, 3)[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
        x2 = self.net2(x2.view(-1, 36))

        x3 = x.view(-1, 33, 3)[:, [11, 12, 14, 16, 18, 20, 22]]
        # x3 = x.view(-1, 33, 3)[:, [11, 12, 13, 15, 17, 19, 21]]
        x3 = self.net3(x3.view(-1, 21))

        x4 = x.view(-1, 33, 3)[:, [11, 12, 13, 15, 17, 19, 21]]
        # x4 = x.view(-1, 33, 3)[:, [11, 12, 14, 16, 18, 20, 22]]
        x4 = self.net4(x4.view(-1, 21))

        x5 = x.view(-1, 33, 3)[:, [23, 24, 26, 28, 30, 32]]
        # x5 = x.view(-1, 33, 3)[:, [23, 24, 25, 27, 29, 31]]
        x5 = self.net5(x5.view(-1, 18))

        x6 = x.view(-1, 33, 3)[:, [23, 24, 25, 27, 29, 31]]
        # x6 = x.view(-1, 33, 3)[:, [23, 24, 26, 28, 30, 32]]
        x6 = self.net6(x6.view(-1, 18))

        return x1, x2, x3, x4, x5, x6

        # # concatenate the outputs
        # x = torch.cat((x1, x2, x3, x4, x5, x6), 1)

        # # Reshape to (batch_size, 22, 3)
        # x = x.reshape(-1, 22, 3)

        # return x


if __name__ == "__main__":

    pass
