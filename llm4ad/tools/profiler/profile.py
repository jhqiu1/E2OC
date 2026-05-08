from __future__ import annotations

import os
import re
import sys
from typing import Literal, Optional, List, Tuple

import numpy as np
import pytz
import json
import logging
from threading import Lock
from datetime import datetime

from ...base import Function


class ProfilerBase:
    """Base profiler for recording experimental results."""

    def __init__(
        self,
        log_dir: Optional[str] = None,
        *,
        initial_num_samples=0,
        log_style: Literal["simple", "complex"] = "complex",
        create_random_path=True,
        num_objs=1,
        **kwargs,
    ):
        assert log_style in ["simple", "complex"]
        self._num_objs = num_objs
        self._num_samples = initial_num_samples
        self._process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
        self._result_folder = self._process_start_time.strftime("%Y%m%d_%H%M%S")

        self._log_dir = log_dir
        self._log_style = log_style
        self._cur_best_function = (
            None if self._num_objs < 2 else [None for _ in range(self._num_objs)]
        )
        self._cur_best_program_sample_order = (
            None if self._num_objs < 2 else [None for _ in range(self._num_objs)]
        )
        self._cur_best_program_score = (
            float("-inf") if self._num_objs < 2 else [float("-inf") for _ in range(self._num_objs)]
        )
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0

        self._parameters = None
        self._logger_txt = logging.getLogger("root")

        if create_random_path:
            self._log_dir = os.path.join(log_dir, self._result_folder)
        else:
            self._log_dir = log_dir

        self._register_function_lock = Lock()

    def record_parameters(self, llm, prob, method):
        self._parameters = [llm, prob, method]
        self._create_log_path()

    def register_function(self, function: Function, program: str = "", *, resume_mode=False):
        if self._num_objs < 2:
            try:
                self._register_function_lock.acquire()
                self._num_samples += 1
                self._record_and_print_verbose(function, resume_mode=resume_mode)
                if not resume_mode:
                    self._write_json(function, program)
            finally:
                self._register_function_lock.release()
        else:
            try:
                self._register_function_lock.acquire()
                self._num_samples += 1
                self._record_and_print_verbose(function, resume_mode=resume_mode)
                if not resume_mode:
                    self._write_json(function, program)
            finally:
                self._register_function_lock.release()

    def finish(self):
        pass

    def _write_json(self, function: Function, program: str, *, record_type="history", record_sep=200):
        if not self._log_dir:
            return

        sample_order = self._num_samples
        content = {
            "sample_order": sample_order,
            "algorithm": getattr(function, "algorithm", ""),
            "function": str(function),
            "score": function.score,
            "program": program,
        }

        if record_type == "history":
            lower_bound = ((sample_order - 1) // record_sep) * record_sep
            upper_bound = lower_bound + record_sep
            filename = f"samples_{lower_bound + 1}~{upper_bound}.json"
        else:
            filename = "samples_best.json"

        path = os.path.join(self._samples_json_dir, filename)
        try:
            with open(path, "r") as json_file:
                data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append(content)
        with open(path, "w") as json_file:
            json.dump(data, json_file, indent=4)

    def _record_and_print_verbose(self, function, program="", *, resume_mode=False):
        function_str = str(function).strip("\n")
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score

        if self._num_objs < 2:
            if score is not None and score > self._cur_best_program_score:
                self._cur_best_function = function
                self._cur_best_program_score = score
                self._cur_best_program_sample_order = self._num_samples
                if not resume_mode:
                    self._write_json(function, record_type="best", program=program)
        else:
            if score is not None:
                for i in range(self._num_objs):
                    if score[i] > self._cur_best_program_score[i]:
                        self._cur_best_function[i] = function
                        self._cur_best_program_score[i] = score[i]
                        self._cur_best_program_sample_order[i] = self._num_samples
                        if not resume_mode:
                            self._write_json(function, record_type="best", program=program)

        if not resume_mode:
            if self._log_style == "complex":
                print(f"================= Evaluated Function =================")
                print(f"{function_str}")
                print(f"------------------------------------------------------")
                print(f"Score        : {str(score)}")
                print(f"Sample time  : {str(sample_time)}")
                print(f"Evaluate time: {str(evaluate_time)}")
                print(f"Sample orders: {str(self._num_samples)}")
                print(f"------------------------------------------------------")
                print(f"Current best score: {self._cur_best_program_score}")
                print(f"======================================================\n")
            else:
                if score is None:
                    if self._num_objs < 2:
                        print(f"Sample{self._num_samples}: Score=None    Cur_Best_Score={self._cur_best_program_score: .3f}")
                    else:
                        print(f"Sample{self._num_samples}: Score=None    Cur_Best_Score=[{self._cur_best_program_score[0]: .3f}, {self._cur_best_program_score[1]: .3f}]")
                else:
                    if self._num_objs < 2:
                        print(f"Sample{self._num_samples}: Score={score: .3f}     Cur_Best_Score={self._cur_best_program_score: .3f}")
                    else:
                        print(f"Sample{self._num_samples}: Score=[{score[0]: .3f}, {score[1]: .3f}]     Cur_Best_Score=[{self._cur_best_program_score[0]: .3f}, {self._cur_best_program_score[1]: .3f}]")

        if score is not None:
            self._evaluate_success_program_num += 1
        else:
             self._evaluate_failed_program_num += 1

        if sample_time is not None:
            self._tot_sample_time += sample_time
        if evaluate_time:
            self._tot_evaluate_time += evaluate_time

    def _create_log_path(self):
        self._samples_json_dir = os.path.join(self._log_dir, "samples")
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(self._samples_json_dir, exist_ok=True)

        file_name = self._log_dir + "/run_log.txt"
        file_mode = "a" if os.path.isfile(file_name) else "w"

        self._logger_txt.setLevel(level=logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        for hdlr in self._logger_txt.handlers[:]:
            self._logger_txt.removeHandler(hdlr)
        fileout = logging.FileHandler(file_name, mode=file_mode)
        fileout.setLevel(logging.INFO)
        fileout.setFormatter(formatter)
        self._logger_txt.addHandler(fileout)
        self._logger_txt.addHandler(logging.StreamHandler(sys.stdout))

    @classmethod
    def load_logfile(cls, logdir, valid_only=False) -> Tuple[List[str], List[float]]:
        file_dir = os.path.join(logdir, "samples")
        all_files = os.listdir(file_dir)
        sample_files = [f for f in all_files if f.startswith("samples_") and f != "samples_best.json"]

        def extract_number(filename):
            match = re.search(r"samples_(\d+)~", filename)
            if match:
                return int(match.group(1))
            return 0

        sorted_files = sorted(sample_files, key=extract_number)
        all_func = []
        all_score = []
        for file in sorted_files:
            file_path = os.path.join(file_dir, file)
            with open(file_path, "r", encoding="utf-8") as f:
                samples = json.load(f)
            for sample in samples:
                func = sample["function"]
                acc = sample["score"] if sample["score"] else float("-inf")
                if valid_only:
                    if acc is None or np.isinf(acc):
                        continue
                all_func.append(func)
                all_score.append(acc)
        return all_func, all_score
