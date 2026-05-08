from __future__ import annotations

import sys
from typing import Optional, Literal

from ...base import Function
from .profile import ProfilerBase

try:
    import wandb
except:
    pass


class WandBProfiler(ProfilerBase):
    def __init__(self, wandb_project_name: str, log_dir: Optional[str] = None, *,
                 initial_num_samples=0, log_style="complex", create_random_path=True,
                 fork_proc: Literal["auto"] | bool = "auto", **wandb_init_kwargs):
        super().__init__(log_dir=log_dir, initial_num_samples=initial_num_samples,
                         log_style=log_style, create_random_path=create_random_path, **wandb_init_kwargs)
        self._wandb_project_name = wandb_project_name

        if fork_proc == "auto":
            if sys.platform.startswith("darwin") or sys.platform.startswith("linux"):
                setting = wandb.Settings(start_method="fork")
                self._logger_wandb = wandb.init(project=self._wandb_project_name,
                                                dir=self._log_dir, settings=setting, **wandb_init_kwargs)
            else:
                wandb.setup()
                self._logger_wandb = wandb.init(project=self._wandb_project_name,
                                                dir=self._log_dir, **wandb_init_kwargs)
        elif fork_proc is True:
            setting = wandb.Settings(start_method="fork")
            self._logger_wandb = wandb.init(project=self._wandb_project_name,
                                            dir=self._log_dir, settings=setting, **wandb_init_kwargs)
        else:
            wandb.setup()
            self._logger_wandb = wandb.init(project=self._wandb_project_name,
                                            dir=self._log_dir, **wandb_init_kwargs)

    def get_logger(self):
        return self._logger_wandb

    def register_function(self, function: Function, program="", *, resume_mode=False):
        try:
            self._register_function_lock.acquire()
            self._num_samples += 1
            self._record_and_print_verbose(function, resume_mode=resume_mode)
            self._write_wandb()
            self._write_json(function, program=program)
        finally:
            self._register_function_lock.release()

    def _write_wandb(self, *args, **kwargs):
        self._logger_wandb.log({"Best Score of Function": self._cur_best_program_score},
                               step=self._num_samples)
        self._logger_wandb.log({
            "Valid Function Num": self._evaluate_success_program_num,
            "Invalid Function Num": self._evaluate_failed_program_num,
        }, step=self._num_samples)
        self._logger_wandb.log({
            "Total Sample Time": self._tot_sample_time,
            "Total Evaluate Time": self._tot_evaluate_time,
        }, step=self._num_samples)

    def finish(self):
        wandb.finish()
