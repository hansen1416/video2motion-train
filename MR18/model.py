import numpy as np
import torch
from torch import Tensor
from torch import nn


class WeightedCombineLayer(nn.Module):
    def __init__(self):
        super(WeightedCombineLayer, self).__init__()
        # Initialize weights with 0.5
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize weights with 0.5
        self.weight_inp1 = torch.nn.Parameter(torch.full((1, 66), 0.5)).to(device)
        self.weight_inp2 = torch.nn.Parameter(torch.full((1, 66), 0.5)).to(device)

    def forward(self, input1, input2):
        return input1 * self.weight_inp2 + input2 * self.weight_inp2


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()

        self.res_dp1 = nn.Dropout(0.25)
        self.res_fc1 = nn.Linear(512, 256)
        self.res_dp2 = nn.Dropout(0.25)
        self.res_fc2 = nn.Linear(256, 128)
        self.res_dp3 = nn.Dropout(0.25)
        self.res_fc3 = nn.Linear(128, 66)

        self.mp_fc1 = nn.Linear(99, 128)
        self.mp_fc2 = nn.Linear(128, 256)
        self.mp_fc3 = nn.Linear(256, 128)
        self.mp_fc4 = nn.Linear(128, 66)

    def forward(self, mediapipe_input, resnet_input):

        mediapipe_input = self.mp_fc1(mediapipe_input)
        mediapipe_input = nn.functional.tanh(mediapipe_input)
        mediapipe_input = self.mp_fc2(mediapipe_input)
        mediapipe_input = nn.functional.tanh(mediapipe_input)
        mediapipe_input = self.mp_fc3(mediapipe_input)
        mediapipe_input = nn.functional.tanh(mediapipe_input)
        mediapipe_input = self.mp_fc4(mediapipe_input)
        # now the shape pf mediapipe_input is (batch_size, 66)

        resnet_input = self.res_dp1(resnet_input)
        resnet_input = self.res_fc1(resnet_input)
        resnet_input = nn.functional.tanh(resnet_input)
        resnet_input = self.res_dp2(resnet_input)
        resnet_input = self.res_fc2(resnet_input)
        resnet_input = nn.functional.tanh(resnet_input)
        resnet_input = self.res_dp3(resnet_input)
        resnet_input = self.res_fc3(resnet_input)
        # clamp the values to be between -pi and pi
        resnet_input = torch.clamp(resnet_input, -np.pi, np.pi)
        # now the shape of resnet_input is (batch_size, 66)

        # apply the weighted combine layer, the output shape is (batch_size, 66)
        combined_output = WeightedCombineLayer()(mediapipe_input, resnet_input)

        # Reshape to (batch_size, 22, 3), so that we can use it as input to the loss function
        output = combined_output.reshape(-1, 22, 3)

        return output
