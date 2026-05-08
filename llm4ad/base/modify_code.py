from __future__ import annotations

import ast
import io
import tokenize
from collections.abc import Iterator, MutableSet
from typing import Sequence, Tuple, List, Dict, Any


class ModifyCode:
    """Static helper class for modifying Python source code."""

    @classmethod
    def add_numpy_random_seed_to_func(cls, program: str, func_name: str, seed: int = 2024) -> str:
        tree = ast.parse(program)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                node.body = [ast.parse(f"np.random.seed({seed})").body[0]] + node.body
        return ast.unparse(tree)

    @classmethod
    def replace_div_with_protected_div(
        cls, program: str, delta: float = 1e-5, numba_accelerate: bool = False
    ) -> str:
        protected_div_str = f"""
def _protected_div(x, y, delta={delta}):
    return x / (y + delta)
        """
        tree = ast.parse(program)
        transformer = _CustomDivisionTransformer("_protected_div")
        modified_tree = transformer.visit(tree)
        modified_code = ast.unparse(modified_tree)
        modified_code = "\n".join([modified_code, "", protected_div_str])
        if numba_accelerate:
            modified_code = cls.add_numba_decorator(modified_code, "_protected_div")
        return modified_code

    @classmethod
    def add_numba_decorator(cls, program: str, function_name: str) -> str:
        tree = ast.parse(program)

        numba_imported = False
        for node in tree.body:
            if isinstance(node, ast.Import) and any(
                alias.name == "numba" for alias in node.names
            ):
                numba_imported = True
                break

        if not numba_imported:
            import_node = ast.Import(names=[ast.alias(name="numba", asname=None)])
            tree.body.insert(0, import_node)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                decorator = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="numba", ctx=ast.Load()),
                        attr="jit",
                        ctx=ast.Load(),
                    ),
                    args=[],
                    keywords=[ast.keyword(arg="nopython", value=ast.Constant(value=True))],
                )
                node.decorator_list.append(decorator)

        return ast.unparse(tree)

    @classmethod
    def rename_function(cls, code: str, source_name: str, target_name: str) -> str:
        if source_name not in code:
            return code
        modified_tokens = []
        for token, is_call in _yield_token_and_is_call(code):
            if is_call and token.string == source_name:
                modified_token = tokenize.TokenInfo(
                    type=token.type, string=target_name,
                    start=token.start, end=token.end, line=token.line,
                )
                modified_tokens.append(modified_token)
            else:
                modified_tokens.append(token)
        return _untokenize(modified_tokens)


def _yield_token_and_is_call(code: str) -> Iterator[tuple[tokenize.TokenInfo, bool]]:
    try:
        tokens = _tokenize(code)
        prev_token = None
        is_attribute_access = False
        for token in tokens:
            if (
                prev_token
                and prev_token.type == tokenize.NAME
                and token.type == tokenize.OP
                and token.string == "("
            ):
                yield prev_token, not is_attribute_access
                is_attribute_access = False
            else:
                if prev_token:
                    is_attribute_access = (
                        prev_token.type == tokenize.OP and prev_token.string == "."
                    )
                    yield prev_token, False
            prev_token = token
        if prev_token:
            yield prev_token, False
    except Exception as e:
        raise e


def _tokenize(code: str) -> Iterator[tokenize.TokenInfo]:
    code_bytes = code.encode()
    code_io = io.BytesIO(code_bytes)
    return tokenize.tokenize(code_io.readline)


def _untokenize(tokens: Sequence[tokenize.TokenInfo]) -> str:
    code_bytes = tokenize.untokenize(tokens)
    return code_bytes.decode()


class _CustomDivisionTransformer(ast.NodeTransformer):
    def __init__(self, custom_divide_func_name: str):
        super().__init__()
        self._custom_div_func = custom_divide_func_name

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Div):
            custom_divide_call = ast.Call(
                func=ast.Name(id=self._custom_div_func, ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[],
            )
            return custom_divide_call
        return node
