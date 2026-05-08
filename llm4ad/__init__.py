from . import base

__version__ = '1.0.0'


def _get_method_eoh():
    from .method import eoh
    return eoh


def _get_tools_llm():
    from .tools import llm
    return llm


def _get_tools_profiler():
    from .tools import profiler
    return profiler
