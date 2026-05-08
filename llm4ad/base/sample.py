from __future__ import annotations

import ast
import copy
from abc import abstractmethod
from typing import Any, List

from .code import Program, Function, TextFunctionProgramConverter


class LLM:
    """Language model that predicts continuation of provided source code."""

    def __init__(self, *, do_auto_trim=True, debug_mode=False):
        self.do_auto_trim = do_auto_trim
        self.debug_mode = debug_mode

    @abstractmethod
    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        """Returns a predicted continuation of `prompt`."""
        pass

    def draw_samples(self, prompts: List[str | Any], *args, **kwargs) -> List[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self.draw_sample(p, *args, **kwargs) for p in prompts]


class SampleTrimmer:
    def __init__(self, llm: LLM):
        self._llm = llm

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        generated_code = self._llm.draw_sample(prompt, *args, **kwargs)
        if self._llm.do_auto_trim:
            generated_code = self.__class__.auto_trim(generated_code)
        return generated_code

    @classmethod
    def auto_trim(cls, generated_code: str) -> str:
        is_code_complete = cls._check_indent_if_code_completion(generated_code)
        if is_code_complete:
            return generated_code
        return cls.trim_preface_of_function(generated_code)

    @classmethod
    def _check_indent_if_code_completion(cls, generated_code: str) -> bool:
        generated_code = generated_code.strip("\n")
        line = generated_code.splitlines()[0]
        if line.startswith("\t") or line.startswith(" "):
            return True
        return False

    @classmethod
    def trim_preface_of_function(cls, generated_code: str):
        lines = generated_code.splitlines()
        func_body_lineno = 0
        find_def_declaration = False
        for lineno, line in enumerate(lines):
            if line[:3] == "def":
                func_body_lineno = lineno
                find_def_declaration = True
                break
        if find_def_declaration:
            code = ""
            for line in lines[func_body_lineno + 1 :]:
                code += line + "\n"
            return code
        return generated_code

    @classmethod
    def sample_to_function(
        cls, generated_code: str, template_program: str | Program
    ) -> Function | None:
        program = cls.sample_to_program(generated_code, template_program)
        if program is None:
            return None
        return TextFunctionProgramConverter.program_to_function(program)

    @classmethod
    def sample_to_program(
        cls, generated_code: str, template_program: str | Program
    ) -> Program | None:
        try:
            generated_code = cls.trim_function_body(generated_code)
            if isinstance(template_program, str):
                template_program = TextFunctionProgramConverter.text_to_program(template_program)
            else:
                template_program = copy.deepcopy(template_program)
            docstr_copy = template_program.functions[0].docstring
            template_program.functions[0].body = generated_code
            template_program.functions[0] = cls.remove_docstrings(template_program.functions[0])
            if (
                template_program.functions[0].body == ""
                or template_program.functions[0].body is None
            ):
                return None
            template_program.functions[0].docstring = docstr_copy
            return template_program
        except ValueError as value_err:
            raise value_err
        except:
            return None

    @classmethod
    def trim_function_body(cls, generated_code: str) -> str | None:
        try:
            if not generated_code:
                return ""
            lines = generated_code.split("\n")
            cleaned_lines = [
                line for line in lines
                if line.strip() not in ("```", "```python", "```py")
            ]
            cleaned_code = "\n".join(cleaned_lines)
            code = f"def fake_function_header():\n{cleaned_code}"

            tree = None
            max_attempts = 10
            attempts = 0

            while tree is None and attempts < max_attempts:
                try:
                    tree = ast.parse(code)
                except SyntaxError as e:
                    lines = code.splitlines()
                    if e.lineno < len(lines):
                        del lines[e.lineno - 1]
                        code = "\n".join(lines)
                    else:
                        code = "\n".join(lines[:-1])
                    attempts += 1

            if tree is None:
                return ""

            visitor = _FunctionLineVisitor("fake_function_header")
            visitor.visit(tree)
            body_lines = code.splitlines()[1 : visitor.function_end_line]
            return "\n".join(body_lines) + "\n\n"
        except Exception as e:
            print(f"Error trimming function body: {e}")
            return None

    @classmethod
    def remove_docstrings(cls, func: Function | str):
        func_ = copy.deepcopy(func)
        func_ = TextFunctionProgramConverter.text_to_function(str(func_))
        docstring = func_.docstring
        while not (docstring == "" or docstring is None):
            func_.docstring = ""
            func_str = str(func_)
            func_ = TextFunctionProgramConverter.text_to_function(func_str)
            docstring = func_.docstring
        if isinstance(func, Function):
            for key, value in func.__dict__.items():
                if key != "docstring" and key != "body":
                    setattr(func_, key, value)
            return func_
        else:
            return str(func_)


class _FunctionLineVisitor(ast.NodeVisitor):
    def __init__(self, target_function_name: str) -> None:
        self._target_function_name: str = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node: Any) -> None:
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        assert self._function_end_line is not None
        return self._function_end_line
