from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader

from data_utils import map_to_tensor, prepare_data
from fsdl.lab01.text_recognizer.data.util import BaseDataset
from fsdl.lab01.text_recognizer.models.mlp import MLP
from nn_1layer_v2_pytorch import configure_opt  # , MNISTLosgistic
