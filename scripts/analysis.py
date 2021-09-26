from __future__ import print_function

import argparse
import json
import os
import re
import traceback


def parse_args():
    """
    爬取从run_all.sh传递过来的参数
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filename", type=str, help="The name of log which need to analysis.")
    # parser.add_argument(
    #     "--log_with_profiler", type=str, help="The path of train log with profiler")
    # parser.add_argument(
    #     "--profiler_path", type=str, help="The path of profiler timeline log.")
    parser.add_argument(
        "--keyword", type=str, help="Keyword to specify analysis data")
    # parser.add_argument(
    #     "--separator", type=str, default=None, help="Separator of different field in log")
    # parser.add_argument(
    #     '--position', type=int, default=None, help='The position of data field')
    # parser.add_argument(
    #     '--range', type=str, default="", help='The range of data field to intercept')
    # parser.add_argument(
    #     '--base_batch_size', type=int, help='base_batch size on gpu')
    # parser.add_argument(
    #     '--skip_steps', type=int, default=0, help='The number of steps to be skipped')
    # parser.add_argument(
    #     '--model_mode', type=int, default=-1, help='Analysis mode, default value is -1')
    # parser.add_argument(
    #     '--ips_unit', type=str, default=None, help='IPS unit')
    parser.add_argument(
        '--model_name', type=str, default=0, help='training model_name, transformer_base')
    parser.add_argument(
        '--mission_name', type=str, default="action recognition", help='training mission name')
    parser.add_argument(
        '--direction_id', type=int, default=0, help='training direction_id')
    parser.add_argument(
        '--run_mode', type=str, default="sp", help='multi process or single process')
    parser.add_argument(
        '--index', type=int, default=1, help='{1: speed, 2:mem, 3:profiler, 6:max_batch_size}')
    parser.add_argument(
        '--gpu_num', type=int, default=1, help='nums of training gpus')
    args = parser.parse_args()
    # args.separator = None if args.separator == "None" else args.separator
    return args


def parse_text_from_file(file_path: str):
    with open(file_path, "r") as f:
        lines = f.read().splitlines()
    return lines


def parse_avgips_from_text(text: list, keyword: str):
    # print(keyword)
    skip_iter = 4
    count_list = []
    for i, line in enumerate(text):
        if i < skip_iter:
            continue
        if keyword in line:
            # print(line)
            words = line.split(" ")
            for j, word in enumerate(words):
                if word == keyword:
                    count_list.append(float(words[j+1]))
                    break
    if count_list:
        return sum(count_list) / len(count_list)
    else:
        return 0.0


if __name__ == '__main__':
    args = parse_args()
    # print(args.keyword)
    run_info = dict()
    run_info["log_file"] = args.filename
    run_info["model_name"] = args.model_name
    run_info["mission_name"] = args.mission_name
    run_info["direction_id"] = args.direction_id
    run_info["run_mode"] = args.run_mode
    run_info["index"] = args.index
    run_info["gpu_num"] = args.gpu_num
    run_info["FINAL_RESULT"] = 0
    run_info["JOB_FAIL_FLAG"] = 0
    # print(args.keyword)
    text = parse_text_from_file(args.filename)
    avg_ips = parse_avgips_from_text(text, args.keyword)
    run_info["FINAL_RESULT"] = avg_ips
    if avg_ips == 0.0:
        run_info["JOB_FAIL_FLAG"] = 1
    print("{}".format(json.dumps(run_info)))
