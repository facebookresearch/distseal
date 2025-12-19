# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import List, Tuple


def parse_dataset_params(args: List[str]) -> Tuple[List[str], List[str]]:
    prefix = "dataset"
    seen = []
    unseen = []
    args_iter = iter(args)

    # param1, value1, param2, value2,... -> (param1, value1), (param2, value2)
    for param, value in zip(args_iter, args_iter):
        _param = param[2:]
        if _param.startswith(prefix):
            # Example arguments conversion: --dataset.name ==> --name
            seen.append(param[:2] + _param[len(prefix) :])
            seen.append(value)
        else:
            unseen.append(param)
            unseen.append(value)
    
    # At this point we don't allow other free params than dataset params
    # We might reuse this function later if we extend to e.g. model free params, training free params, etc.
    return seen, unseen
