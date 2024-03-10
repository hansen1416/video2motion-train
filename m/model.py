from typing import Tuple
from torch import nn, Tensor

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

"""


class Model(nn.Module):
    def __init__(self) -> None:

        super(Model, self).__init__()
        # [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28] => [0, 1, 2, 3]
        self.hip_spine = nn.Sequential(
            nn.Linear(36, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 12),
        )
        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] => [4, 5]
        self.neck_head = nn.Sequential(
            nn.Linear(36, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 6),
        )

        # [11, 12, 14, 16, 18, 20, 22, 23, 24] => [6, 7, 8]
        self.right_arm = nn.Sequential(
            nn.Linear(27, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 9),
        )

        # [11, 12, 14, 16, 18, 20, 22, 23, 24] => [9]
        self.right_hand = nn.Sequential(
            nn.Linear(27, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 3),
        )

        # [11, 12, 13, 15, 17, 19, 21, 23, 24] => [10, 11, 12]
        self.left_arm = nn.Sequential(
            nn.Linear(27, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 9),
        )

        # [11, 12, 13, 15, 17, 19, 21, 23, 24] => [13]
        self.left_hand = nn.Sequential(
            nn.Linear(27, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 16),
            nn.Tanh(),
            nn.Linear(16, 3),
        )

        # [11, 12, 23, 24, 26, 28, 30, 32] => [14, 15]
        self.right_leg = nn.Sequential(
            nn.Linear(24, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 6),
        )

        # [11, 12, 23, 24, 26, 28, 30, 32] => [16, 17]
        self.right_foot = nn.Sequential(
            nn.Linear(24, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 6),
        )

        # [11, 12, 23, 24, 25, 27, 29, 31] => [18, 19]
        self.left_leg = nn.Sequential(
            nn.Linear(24, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 6),
        )

        # [11, 12, 23, 24, 25, 27, 29, 31] => [20, 21]
        self.left_foot = nn.Sequential(
            nn.Linear(24, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 6),
        )

    def forward(
        self, x
    ) -> Tuple[
        Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor
    ]:

        x = x.view(-1, 33, 3)

        # [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28] => [0, 1, 2, 3]
        # include joints 11: "LEFT_SHOULDER",12: "RIGHT_SHOULDER",
        # 13: "LEFT_ELBOW",14: "RIGHT_ELBOW",
        # 15: "LEFT_WRIST",16: "RIGHT_WRIST",
        # 23: "LEFT_HIP",24: "RIGHT_HIP",
        # 25: "LEFT_KNEE",26: "RIGHT_KNEE",
        # 27: "LEFT_ANKLE",28: "RIGHT_ANKLE",
        # to predict 0: "Hips", 1: "Spine", 2: "Spine1", 3: "Spine2",
        x1 = x[:, [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]]
        x1 = self.hip_spine(x1.view(-1, 36))

        # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] => [4, 5]
        # include joints 1: "LEFT_EYE_INNER",2: "LEFT_EYE",3: "LEFT_EYE_OUTER",
        # 4: "RIGHT_EYE_INNER",5: "RIGHT_EYE",6: "RIGHT_EYE_OUTER",
        # 7: "LEFT_EAR",8: "RIGHT_EAR", 9: "LEFT_RIGHT",10: "RIGHT_LEFT",
        # 11: "LEFT_SHOULDER",12: "RIGHT_SHOULDER",
        # to predict 4: "Neck", 5: "Head",
        x2 = x[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
        x2 = self.neck_head(x2.view(-1, 36))

        # [11, 12, 14, 16, 18, 20, 22, 23, 24] => [6, 7, 8]
        # include joints 11: "LEFT_SHOULDER",12: "RIGHT_SHOULDER",
        # 14: "RIGHT_ELBOW",16: "RIGHT_WRIST",
        # 18: "RIGHT_PINKY",20: "RIGHT_INDEX", 22: "RIGHT_THUMB",
        # 23: "LEFT_HIP",24: "RIGHT_HIP",
        # to predict 6: "RightShoulder", 7: "RightArm", 8: "RightForeArm",
        x3 = x[:, [11, 12, 14, 16, 18, 20, 22, 23, 24]]
        x3 = self.right_arm(x3.view(-1, 27))

        # [11, 12, 14, 16, 18, 20, 22, 23, 24] => [9]
        # include joints 11: "LEFT_SHOULDER",12: "RIGHT_SHOULDER",
        # 14: "RIGHT_ELBOW",16: "RIGHT_WRIST",
        # 18: "RIGHT_PINKY",20: "RIGHT_INDEX", 22: "RIGHT_THUMB",
        # 23: "LEFT_HIP",24: "RIGHT_HIP",
        # to predict 9: "RightHand",
        x4 = x[:, [11, 12, 14, 16, 18, 20, 22, 23, 24]]
        x4 = self.right_hand(x4.view(-1, 27))

        # [11, 12, 13, 15, 17, 19, 21, 23, 24] => [10, 11, 12]
        # include joints 11: "LEFT_SHOULDER",12: "RIGHT_SHOULDER",
        # 13: "LEFT_ELBOW",15: "LEFT_WRIST",
        # 17: "LEFT_PINKY",19: "LEFT_INDEX", 21: "LEFT_THUMB",
        # 23: "LEFT_HIP",24: "RIGHT_HIP",
        # to predict 10: "LeftShoulder", 11: "LeftArm", 12: "LeftForeArm",
        x5 = x[:, [11, 12, 13, 15, 17, 19, 21, 23, 24]]
        x5 = self.left_arm(x5.view(-1, 27))

        # [11, 12, 13, 15, 17, 19, 21, 23, 24] => [13]
        # include joints 11: "LEFT_SHOULDER",12: "RIGHT_SHOULDER",
        # 13: "LEFT_ELBOW",15: "LEFT_WRIST",
        # 17: "LEFT_PINKY",19: "LEFT_INDEX", 21: "LEFT_THUMB",
        # 23: "LEFT_HIP",24: "RIGHT_HIP",
        # to predict 13: "LeftHand",
        x6 = x[:, [11, 12, 13, 15, 17, 19, 21, 23, 24]]
        # x6 = x.view(-1, 33, 3)[:, [23, 24, 26, 28, 30, 32]]
        x6 = self.left_hand(x6.view(-1, 27))

        # [11, 12, 23, 24, 26, 28, 30, 32] => [14, 15]
        # include joints 11: "LEFT_SHOULDER",12: "RIGHT_SHOULDER",
        # 23: "LEFT_HIP",24: "RIGHT_HIP",
        # 26: "RIGHT_KNEE",28: "RIGHT_ANKLE",
        # 30: "RIGHT_HEEL",32: "RIGHT_FOOT_INDEX",
        # to predict 14: "RightUpLeg", 15: "RightLeg",
        x7 = x[:, [11, 12, 23, 24, 26, 28, 30, 32]]
        x7 = self.right_leg(x7.view(-1, 24))

        # [11, 12, 23, 24, 26, 28, 30, 32] => [16, 17]
        # include joints 11: "LEFT_SHOULDER",12: "RIGHT_SHOULDER",
        # 23: "LEFT_HIP",24: "RIGHT_HIP",
        # 26: "RIGHT_KNEE",28: "RIGHT_ANKLE",
        # 30: "RIGHT_HEEL",32: "RIGHT_FOOT_INDEX",
        # to predict 16: "RightFoot", 17: "RightToeBase",
        x8 = x[:, [11, 12, 23, 24, 26, 28, 30, 32]]
        x8 = self.right_foot(x8.view(-1, 24))

        # [11, 12, 23, 24, 25, 27, 29, 31] => [18, 19]
        # include joints 11: "LEFT_SHOULDER",12: "RIGHT_SHOULDER",
        # 23: "LEFT_HIP",24: "RIGHT_HIP",
        # 25: "LEFT_KNEE",27: "LEFT_ANKLE",
        # 29: "LEFT_HEEL",31: "LEFT_FOOT_INDEX",
        # to predict 18: "LeftUpLeg", 19: "LeftLeg",
        x9 = x[:, [11, 12, 23, 24, 25, 27, 29, 31]]
        x9 = self.left_leg(x9.view(-1, 24))

        # [11, 12, 23, 24, 25, 27, 29, 31] => [20, 21]
        # include joints 11: "LEFT_SHOULDER",12: "RIGHT_SHOULDER",
        # 23: "LEFT_HIP",24: "RIGHT_HIP",
        # 25: "LEFT_KNEE",27: "LEFT_ANKLE",
        # 29: "LEFT_HEEL",31: "LEFT_FOOT_INDEX",
        # to predict 20: "LeftFoot", 21: "LeftToeBase",
        x10 = x[:, [11, 12, 23, 24, 25, 27, 29, 31]]
        x10 = self.left_foot(x10.view(-1, 24))

        return x1, x2, x3, x4, x5, x6, x7, x8, x9, x10

        # # concatenate the outputs
        # x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10), 1)

        # # Reshape to (batch_size, 22, 3)
        # x = x.reshape(-1, 22, 3)

        # return x


if __name__ == "__main__":

    pass
