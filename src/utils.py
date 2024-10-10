import os
import random

import torch
from prettytable import PrettyTable


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class logger(object):
    def __init__(self, path, log_name="log.txt"):
        self.path_to_log = os.path.join(path, log_name)

    def info(self, msg, print_msg=True):
        if print_msg:
            print(msg)
        with open(self.path_to_log, 'a') as f:
            f.write(msg + "\n")


def get_args_table(args_dict):
    table = PrettyTable(['Arg', 'Value'])
    for arg, val in args_dict.items():
        table.add_row([arg, val])
    return table