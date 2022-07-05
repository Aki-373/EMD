import os
import torch

from model import *

dir_path = "./result"
os.makedirs(dir_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model().to(device)