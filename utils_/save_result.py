import datetime
import os
import json
from argparse import Namespace
import torch
import numpy as np


def save_result_args(result_content, args,
                     file_path="./results/logs.txt",
                     mode='auto'):
    """
    result_content : str/dict
    args : Namespace/dict
    file_path : str           
    mode : str
    """
    if isinstance(args, Namespace):
        args_dict_original = vars(args)
    elif isinstance(args, dict):
        args_dict_original = args.copy()
    else:
        args_dict_original = {"args_original_type": str(type(args)), "args_value": str(args)}

    args_dict_serializable = {}
    for key, value in args_dict_original.items():
        if isinstance(value, torch.device):
            args_dict_serializable[key] = str(value)
        elif isinstance(value, torch.Tensor):

            args_dict_serializable[key] = f"<Tensor shape={value.shape} dtype={value.dtype}>"
        elif isinstance(value, np.ndarray):

            args_dict_serializable[key] = f"<ndarray shape={value.shape} dtype={value.dtype}>"

        elif hasattr(value, '__dict__') and not isinstance(value,
                                                           (str, int, float, bool, list, dict, tuple, type(None))):

            try:
                args_dict_serializable[key] = f"<Object of type {value.__class__.__name__}>"
            except TypeError:
                args_dict_serializable[key] = f"<Unserializable object of type {value.__class__.__name__}>"
        else:
            args_dict_serializable[key] = value

    if isinstance(result_content, dict):
        result_str = json.dumps(result_content, indent=2, ensure_ascii=False)
    else:
        result_str = str(result_content)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_header = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    final_path = file_path
    if mode == 'new':
        base, ext = os.path.splitext(file_path)
        final_path = f"{base}_{time_header}{ext}"
    file_dir = os.path.dirname(final_path)

    if file_dir and not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)
    elif not file_dir and final_path:
        pass

    content = f"""
{'=' * 40}
[运行时间] {timestamp}
[参数设置]
{json.dumps(args_dict_serializable, indent=2, ensure_ascii=False)}

[运行结果]
{result_str}

"""
    write_mode = 'a' if mode == 'auto' else 'w'
    if mode == 'auto' and not os.path.exists(final_path):
        write_mode = 'w'

    with open(final_path, write_mode, encoding='utf-8') as f:
        f.write(content)
    return os.path.abspath(final_path)
