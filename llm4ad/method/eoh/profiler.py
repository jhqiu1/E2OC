from __future__ import annotations

import json
import os
from threading import Lock
from typing import Optional

try:
    import wandb
except:
    pass

from .population import Population
from ...base import Function
from ...tools.profiler import ProfilerBase

class EoHProfiler(ProfilerBase):
    _cur_gen = 0

    def __init__(self,
                 log_dir: Optional[str] = None,
                 *,
                 evaluation_name='Problem',
                 method_name='EoH',
                 initial_num_samples=0,
                 log_style='complex',
                 create_random_path=True,
                 **kwargs):
        super().__init__(evaluation_name=evaluation_name,
                         method_name=method_name,
                         log_dir=log_dir,
                         initial_num_samples=initial_num_samples,
                         log_style=log_style,
                         create_random_path=create_random_path,
                         **kwargs)
        self._pop_lock = Lock()
        if self._log_dir:
            self._ckpt_dir = os.path.join(self._log_dir, 'population')
            os.makedirs(self._ckpt_dir, exist_ok=True)

    def register_population(self, pop: Population):
        try:
            self._pop_lock.acquire()
            if (self._num_samples == 0 or
                    pop.generation == self.__class__._cur_gen):
                return
            funcs = pop.population
            funcs_json = []
            for f in funcs:
                f_json = {
                    'algorithm': f.algorithm,
                    'function': str(f),
                    'score': f.score
                }
                funcs_json.append(f_json)
            path = os.path.join(self._ckpt_dir, f'pop_{pop.generation}.json')
            with open(path, 'w') as json_file:
                json.dump(funcs_json, json_file, indent=4)
            self.__class__._cur_gen += 1
        finally:
            if self._pop_lock.locked():
                self._pop_lock.release()

    def _write_json(self, function: Function, program: str = "", *, record_type='history', record_sep=200):
        assert record_type in ['history', 'best']
        if not self._log_dir:
            return

        sample_order = self._num_samples
        content = {
            'sample_order': sample_order,
            'algorithm': function.algorithm,
            'function': str(function),
            'score': function.score,
            'program': program,
        }

        if record_type == 'history':
            lower_bound = ((sample_order - 1) // record_sep) * record_sep
            upper_bound = lower_bound + record_sep
            filename = f'samples_{lower_bound + 1}~{upper_bound}.json'
        else:
            filename = 'samples_best.json'

        path = os.path.join(self._samples_json_dir, filename)
        try:
            with open(path, 'r') as json_file:
                data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append(content)
        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)


try:
    from ...tools.profiler.tensorboard_profiler import TensorboardProfiler
    class EoHTensorboardProfiler(TensorboardProfiler, EoHProfiler):
        def __init__(self, log_dir: str | None = None, *, evaluation_name='Problem',
                     method_name='EoH', initial_num_samples=0, log_style='complex',
                     create_random_path=True, **kwargs):
            EoHProfiler.__init__(self, log_dir=log_dir, evaluation_name=evaluation_name,
                                 create_random_path=create_random_path, method_name=method_name, **kwargs)
            TensorboardProfiler.__init__(self, log_dir=log_dir, evaluation_name=evaluation_name,
                                         method_name=method_name, initial_num_samples=initial_num_samples,
                                         log_style=log_style, create_random_path=create_random_path, **kwargs)

        def finish(self):
            if self._log_dir:
                self._writer.close()
            filename = 'end.json'
            path = os.path.join(os.path.join(self._log_dir, 'population'), filename)
            with open(path, 'w') as json_file:
                json.dump([], json_file, indent=4)
except ImportError:
    pass

try:
    from ...tools.profiler.wandb_profiler import WandBProfiler
    class EoHWandbProfiler(WandBProfiler, EoHProfiler):
        _cur_gen = 0

        def __init__(self, wandb_project_name: str, log_dir: str | None = None, *,
                     evaluation_name='Problem', method_name='EoH', initial_num_samples=0,
                     log_style='complex', create_random_path=True, **kwargs):
            EoHProfiler.__init__(self, log_dir=log_dir, evaluation_name=evaluation_name,
                                 create_random_path=create_random_path, method_name=method_name, **kwargs)
            WandBProfiler.__init__(self, wandb_project_name=wandb_project_name, log_dir=log_dir,
                                   evaluation_name=evaluation_name, method_name=method_name,
                                   initial_num_samples=initial_num_samples, log_style=log_style,
                                   create_random_path=create_random_path, **kwargs)
            self._pop_lock = Lock()
            if self._log_dir:
                self._ckpt_dir = os.path.join(self._log_dir, 'population')
                os.makedirs(self._ckpt_dir, exist_ok=True)

        def finish(self):
            wandb.finish()
            filename = 'end.json'
            path = os.path.join(os.path.join(self._log_dir, 'population'), filename)
            with open(path, 'w') as json_file:
                json.dump([], json_file, indent=4)
except ImportError:
    pass
