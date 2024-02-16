# -*- coding: utf-8 -*-

import json
import logging
import argparse


def log_info(s, use_log=True):
    if use_log:
        logging.info(s)
    else:
        print(s)


def write_str_to_txt(file_path, str, mode='a'):
    with open(file_path, mode) as f:
        f.write(str)


def write_namespace_to_txt(file_path, json_str, indent=4):
    with open(file_path, 'a') as f:
        f.write(json.dumps(vars(json_str), indent=indent))
        f.write('\n')


def read_txt_to_str(file_path):
    with open(file_path, 'r') as f:
        info_list = f.read().splitlines()
        return info_list


def read_txt_to_namespace(file_path):
    with open(file_path, 'r') as f:
        json_str = json.load(f)
        args = argparse.Namespace(**json_str)
        if type(args.loss_choice) is str:
            # 向旧版本cfg兼容
            args.loss_choice=[args.loss_choice]
            args.loss_lambda=[1]
            args.loss_return_dict=False
        return args


def replace_txt_str(txt_path, old_str, new_str):
    file_data = ''
    with open(txt_path, 'r') as f:
        for idx, line in enumerate(f):
            if old_str in line:
                line = line.replace(old_str, new_str)
            file_data += line
    with open(txt_path, 'w') as f:
        f.write(file_data)
