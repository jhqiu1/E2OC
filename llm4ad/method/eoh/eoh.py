from __future__ import annotations

import concurrent.futures
import time
import traceback
from threading import Thread
from typing import Optional, Literal

from .population import Population
from .profiler import EoHProfiler
from .prompt import EoHPrompt
from .sampler import EoHSampler
from ...base import (
    Evaluation,
    LLM,
    Function,
    Program,
    TextFunctionProgramConverter,
    SecureEvaluator,
    Generator,
)
from ...tools.profiler import ProfilerBase


class EoH(Generator):
    """Evolution of Heuristics: LLM + Evolutionary Computation for Automatic Algorithm Design."""

    def __init__(
        self,
        llm: LLM,
        evaluation: Evaluation,
        profiler: ProfilerBase = None,
        max_generations: Optional[int] = 10,
        max_sample_nums: Optional[int] = 100,
        pop_size: Optional[int] = 5,
        selection_num=2,
        use_e2_operator: bool = True,
        use_m1_operator: bool = True,
        use_m2_operator: bool = True,
        num_samplers: int = 1,
        num_evaluators: int = 1,
        *,
        resume_mode: bool = False,
        initial_sample_nums_max: int = 50,
        debug_mode: bool = False,
        multi_thread_or_process_eval: Literal["thread", "process"] = "thread",
        **kwargs,
    ):
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._max_generations = max_generations
        self._max_sample_nums = max_sample_nums
        self._pop_size = pop_size
        self._selection_num = selection_num
        self._use_e2_operator = use_e2_operator
        self._use_m1_operator = use_m1_operator
        self._use_m2_operator = use_m2_operator

        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._resume_mode = resume_mode
        self._initial_sample_nums_max = initial_sample_nums_max
        self._debug_mode = debug_mode
        llm.debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval

        self._function_to_evolve: Function = (
            TextFunctionProgramConverter.text_to_function(self._template_program_str)
        )
        if self._function_to_evolve is None:
            raise ValueError(
                f"Failed to parse template program into a function:\n"
                f"{self._template_program_str[:500]}..."
            )
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(
            self._template_program_str
        )

        self._adjust_pop_size()

        self._population = Population(pop_size=self._pop_size)
        self._sampler = EoHSampler(llm, self._template_program_str)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler

        self._tot_sample_nums = 0
        self._initial_sample_nums_max = max(self._initial_sample_nums_max, 2 * self._pop_size)

        assert multi_thread_or_process_eval in ["thread", "process"]
        if multi_thread_or_process_eval == "thread":
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_evaluators
            )

        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)

    def _adjust_pop_size(self):
        if self._max_sample_nums >= 10000:
            if self._pop_size is None:
                self._pop_size = 40
        elif self._max_sample_nums >= 1000:
            if self._pop_size is None:
                self._pop_size = 20
        elif self._max_sample_nums >= 200:
            if self._pop_size is None:
                self._pop_size = 10
        else:
            if self._pop_size is None:
                self._pop_size = 5

    def _sample_evaluate_register(self, prompt):
        sample_start = time.time()
        thought, func = self._sampler.get_thought_and_function(prompt)
        sample_time = time.time() - sample_start
        if thought is None or func is None:
            return
        program = TextFunctionProgramConverter.function_to_program(
            func, self._template_program
        )
        if program is None:
            return
        score, eval_time = self._evaluation_executor.submit(
            self._evaluator.evaluate_program_record_time, program
        ).result()
        func.score = score
        func.evaluate_time = eval_time
        func.algorithm = thought
        func.sample_time = sample_time
        if self._profiler is not None:
            self._profiler.register_function(func)
            if isinstance(self._profiler, EoHProfiler):
                self._profiler.register_population(self._population)
            self._tot_sample_nums += 1
        self._population.register_function(func)

    def _continue_loop(self) -> bool:
        if self._max_generations is None and self._max_sample_nums is None:
            return True
        elif self._max_generations is not None and self._max_sample_nums is None:
            return self._population.generation < self._max_generations
        elif self._max_generations is None and self._max_sample_nums is not None:
            return self._tot_sample_nums < self._max_sample_nums
        else:
            return (
                self._population.generation < self._max_generations
                and self._tot_sample_nums < self._max_sample_nums
            )

    def _iteratively_use_eoh_operator(self):
        while self._continue_loop():
            try:
                indivs = [self._population.selection() for _ in range(self._selection_num)]
                prompt = EoHPrompt.get_prompt_e1(
                    self._task_description_str, indivs, self._function_to_evolve
                )
                if self._debug_mode:
                    print(f"E1 Prompt: {prompt}")
                self._sample_evaluate_register(prompt)
                if not self._continue_loop():
                    break

                if self._use_e2_operator:
                    indivs = [self._population.selection() for _ in range(self._selection_num)]
                    prompt = EoHPrompt.get_prompt_e2(
                        self._task_description_str, indivs, self._function_to_evolve
                    )
                    if self._debug_mode:
                        print(f"E2 Prompt: {prompt}")
                    self._sample_evaluate_register(prompt)
                    if not self._continue_loop():
                        break

                if self._use_m1_operator:
                    indiv = self._population.selection()
                    prompt = EoHPrompt.get_prompt_m1(
                        self._task_description_str, indiv, self._function_to_evolve
                    )
                    if self._debug_mode:
                        print(f"M1 Prompt: {prompt}")
                    self._sample_evaluate_register(prompt)
                    if not self._continue_loop():
                        break

                if self._use_m2_operator:
                    indiv = self._population.selection()
                    prompt = EoHPrompt.get_prompt_m2(
                        self._task_description_str, indiv, self._function_to_evolve
                    )
                    if self._debug_mode:
                        print(f"M2 Prompt: {prompt}")
                    self._sample_evaluate_register(prompt)
                    if not self._continue_loop():
                        break
            except KeyboardInterrupt:
                break
            except Exception as e:
                if self._debug_mode:
                    traceback.print_exc()
                continue

    def _iteratively_init_population(self):
        while self._population.generation == 0:
            try:
                prompt = EoHPrompt.get_prompt_i1(
                    self._task_description_str, self._function_to_evolve
                )
                self._sample_evaluate_register(prompt)
                if self._tot_sample_nums > self._initial_sample_nums_max:
                    print(
                        f"Warning: Initialization not accomplished in {self._initial_sample_nums_max} samples !!!"
                    )
                    break
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                continue

    def _multi_threaded_sampling(self, fn: callable, *args, **kwargs):
        sampler_threads = [
            Thread(target=fn, args=args, kwargs=kwargs)
            for _ in range(self._num_samplers)
        ]
        for t in sampler_threads:
            t.start()
        for t in sampler_threads:
            t.join()

    def run(self):
        if not self._resume_mode:
            self._multi_threaded_sampling(self._iteratively_init_population)
            if len(self._population) < self._selection_num:
                print(
                    f"The search is terminated since EoH unable to obtain {self._selection_num} feasible algorithms during initialization. "
                    f"Please increase the `initial_sample_nums_max` argument (currently {self._initial_sample_nums_max}). "
                    f"Please also check your evaluation implementation and LLM implementation."
                )
                self._shutdown_executor()
                return
        self._multi_threaded_sampling(self._iteratively_use_eoh_operator)
        self._shutdown_executor()
        if self._profiler is not None:
            self._profiler.finish()

    def _shutdown_executor(self):
        try:
            self._evaluation_executor.shutdown(cancel_futures=True)
        except Exception:
            pass

    def get_population(self):
        """Return the final population of generated operators."""
        return self._population
