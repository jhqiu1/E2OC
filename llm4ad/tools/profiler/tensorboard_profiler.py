from __future__ import annotations

import os
from typing import Optional

from ...base import Function
from .profile import ProfilerBase

try:
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    from torch.utils.tensorboard import SummaryWriter
except:
    pass


class TensorboardProfiler(ProfilerBase):
    def __init__(self, log_dir: Optional[str] = None, *, initial_num_samples=0,
                 log_style="complex", create_random_path=True, **kwargs):
        super().__init__(log_dir=log_dir, initial_num_samples=initial_num_samples,
                         log_style=log_style, create_random_path=create_random_path, **kwargs)
        if log_dir:
            self._writer = SummaryWriter(log_dir=self._log_dir)

    def get_logger(self):
        return self._writer

    def register_function(self, function: Function, program="", *, resume_mode=False):
        try:
            self._register_function_lock.acquire()
            self._num_samples += 1
            self._record_and_print_verbose(function, resume_mode=resume_mode)
            self._write_tensorboard()
            self._write_json(function, program=program)
        finally:
            self._register_function_lock.release()

    def finish(self):
        if self._log_dir:
            self._writer.close()

    def _write_tensorboard(self, *args, **kwargs):
        if not self._log_dir:
            return
        self._writer.add_scalar("Best Score of Function", self._cur_best_program_score,
                                global_step=self._num_samples)
        self._writer.add_scalars("Legal/Illegal Function", {
            "legal function num": self._evaluate_success_program_num,
            "illegal function num": self._evaluate_failed_program_num,
        }, global_step=self._num_samples)
        self._writer.add_scalars("Total Sample/Evaluate Time", {
            "sample time": self._tot_sample_time,
            "evaluate time": self._tot_evaluate_time,
        }, global_step=self._num_samples)
