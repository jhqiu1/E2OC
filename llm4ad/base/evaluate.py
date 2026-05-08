from __future__ import annotations

import multiprocessing
import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Literal

from .code import TextFunctionProgramConverter, Program
from .modify_code import ModifyCode


class Evaluation(ABC):
    def __init__(
        self,
        template_program: str | Program,
        task_description: str = "",
        use_numba_accelerate: bool = False,
        use_protected_div: bool = False,
        protected_div_delta: float = 1e-5,
        random_seed: int | None = None,
        timeout_seconds: int | float = None,
        *,
        exec_code: bool = True,
        safe_evaluate: bool = True,
        daemon_eval_process: bool = False,
    ):
        """Evaluation interface for executing generated code.

        Args:
            template_program: The program template string or Program instance.
            task_description: Description of the task.
            use_numba_accelerate: Wrap the function with '@numba.jit(nopython=True)'.
            use_protected_div: Modify 'a / b' => 'a / (b + delta)'.
            protected_div_delta: Delta value in protected div.
            random_seed: If not None, set random seed in the function body.
            timeout_seconds: Terminate evaluation after timeout seconds.
            exec_code: Use exec() to compile code and provide callable function.
            safe_evaluate: Evaluate in safe mode using a new process.
            daemon_eval_process: Set the evaluate process as a daemon process.
        """
        self.template_program = template_program
        self.task_description = task_description
        self.use_numba_accelerate = use_numba_accelerate
        self.use_protected_div = use_protected_div
        self.protected_div_delta = protected_div_delta
        self.random_seed = random_seed
        self.timeout_seconds = timeout_seconds
        self.exec_code = exec_code
        self.safe_evaluate = safe_evaluate
        self.daemon_eval_process = daemon_eval_process

    @abstractmethod
    def evaluate_program(
        self, program_str: str, callable_func: callable, **kwargs
    ) -> Any | None:
        """Evaluate a given function.

        Args:
            program_str: The function in string.
            callable_func: The callable heuristic function.

        Returns:
            The fitness value.
        """
        raise NotImplementedError("Must provide a evaluator for a function.")


class SecureEvaluator:
    def __init__(
        self,
        evaluator: Evaluation,
        debug_mode=False,
        *,
        fork_proc: Literal["auto", "default"] | bool = "auto",
        **kwargs,
    ):
        assert fork_proc in [True, False, "auto", "default"]
        self._evaluator = evaluator
        self._debug_mode = debug_mode

        if self._evaluator.safe_evaluate:
            if fork_proc == "auto":
                if sys.platform.startswith("darwin") or sys.platform.startswith(
                    "linux"
                ):
                    multiprocessing.set_start_method("fork", force=True)
            elif fork_proc is True:
                multiprocessing.set_start_method("fork", force=True)
            elif fork_proc is False:
                multiprocessing.set_start_method("spawn", force=True)

    def _modify_program_code(self, program_str: str) -> str:
        function_name = TextFunctionProgramConverter.text_to_function(program_str).name
        if self._evaluator.use_numba_accelerate:
            program_str = ModifyCode.add_numba_decorator(
                program_str, function_name=function_name
            )
        if self._evaluator.use_protected_div:
            program_str = ModifyCode.replace_div_with_protected_div(
                program_str,
                self._evaluator.protected_div_delta,
                self._evaluator.use_numba_accelerate,
            )
        if self._evaluator.random_seed is not None:
            program_str = ModifyCode.add_numpy_random_seed_to_func(
                program_str, function_name, self._evaluator.random_seed
            )
        return program_str

    def evaluate_program(self, program: str | Program, **kwargs):
        try:
            program_str = str(program)
            function_name = TextFunctionProgramConverter.text_to_function(
                program_str
            ).name

            program_str = self._modify_program_code(program_str)
            if self._debug_mode:
                print(f"DEBUG: evaluated program:\n{program_str}\n")

            if self._evaluator.safe_evaluate:
                result_queue = multiprocessing.Queue()
                process = multiprocessing.Process(
                    target=self._evaluate_in_safe_process,
                    args=(program_str, function_name, result_queue),
                    kwargs=kwargs,
                    daemon=self._evaluator.daemon_eval_process,
                )
                process.start()

                if self._evaluator.timeout_seconds is not None:
                    try:
                        result = result_queue.get(
                            timeout=self._evaluator.timeout_seconds
                        )
                        process.terminate()
                        process.join()
                    except:
                        if self._debug_mode:
                            print(
                                f"DEBUG: the evaluation time exceeds {self._evaluator.timeout_seconds}s."
                            )
                        process.terminate()
                        process.join()
                        result = None
                else:
                    result = result_queue.get()
                    process.terminate()
                    process.join()
                return result
            else:
                return self._evaluate(program_str, function_name, **kwargs)
        except Exception as e:
            if self._debug_mode:
                print(e)
            return None

    def evaluate_program_record_time(self, program: str | Program, **kwargs):
        evaluate_start = time.time()
        result = self.evaluate_program(program, **kwargs)
        return result, time.time() - evaluate_start

    def _evaluate_in_safe_process(
        self,
        program_str: str,
        function_name,
        result_queue: multiprocessing.Queue,
        **kwargs,
    ):
        try:
            if self._evaluator.exec_code:
                all_globals_namespace = {}
                exec(program_str, all_globals_namespace)
                program_callable = all_globals_namespace[function_name]
            else:
                program_callable = None

            res = self._evaluator.evaluate_program(
                program_str, program_callable, **kwargs
            )
            result_queue.put(res)
        except Exception as e:
            if self._debug_mode:
                print(e)
            result_queue.put(None)

    def _evaluate(self, program_str: str, function_name, **kwargs):
        try:
            if self._evaluator.exec_code:
                all_globals_namespace = {}
                exec(program_str, all_globals_namespace)
                program_callable = all_globals_namespace[function_name]
            else:
                program_callable = None

            res = self._evaluator.evaluate_program(
                program_str, program_callable, **kwargs
            )
            return res
        except Exception as e:
            if self._debug_mode:
                print(e)
            return None
