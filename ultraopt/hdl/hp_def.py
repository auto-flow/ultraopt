import math
from typing import Any, List

from ConfigSpace import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    OrdinalHyperparameter

from ultraopt.utils.math_ import float_gcd


def _encode(value: Any) -> str:
    if isinstance(value, str):
        return value
    return f'{value}:{(value).__class__.__name__}'


def _decode(str_value: str) -> Any:
    if str_value == "None":
        return None
    ix = str_value.rfind(":")
    if ix < 0:
        return str_value
    else:
        value_ = str_value[:ix]
        type_ = str_value[ix + 1:]
        try:
            return eval(value_)
        except Exception:
            return str_value


def choice(label: str, options: List, default=None):
    if len(options) == 1:
        return Constant(label, _encode(options[0]))
    choices = []
    for option in options:
        choices.append(_encode(option))
    kwargs = {}
    if default:
        kwargs.update({'default_value': _encode(default)})
    hp = CategoricalHyperparameter(label, choices, **kwargs)
    return hp


def ordinal(label: str, sequence: List, default=None):
    if len(sequence) == 1:
        return Constant(label, _encode(sequence[0]))
    choices = []
    for option in sequence:
        choices.append(_encode(option))
    kwargs = {}
    if default:
        kwargs.update({'default_value': _encode(default)})
    hp = OrdinalHyperparameter(label, choices, **kwargs)
    return hp


def int_quniform(label: str, low: int, high: int, q: int = None, default=None):
    if not q:
        q = math.gcd(low, high)
    kwargs = {}
    if default:
        kwargs.update({'default_value': default})
    return UniformIntegerHyperparameter(label, low, high, q=q, **kwargs)


def int_uniform(label: str, low: int, high: int, default=None):
    kwargs = {}
    if default:
        kwargs.update({'default_value': default})
    return UniformIntegerHyperparameter(label, low, high, **kwargs)


def quniform(label: str, low: float, high: float, q: float = None, default=None):
    if not q:
        q = float_gcd(low, high)
    kwargs = {}
    if default:
        kwargs.update({'default_value': default})
    return UniformFloatHyperparameter(label, low, high, q=q, **kwargs)


def uniform(label: str, low: float, high: float, default=None):
    kwargs = {}
    if default:
        kwargs.update({'default_value': default})
    return UniformFloatHyperparameter(label, low, high, **kwargs)


# fixme: have some bug in practice
def qloguniform(label: str, low: float, high: float, q: float = None, default=None):
    if not q:
        q = float_gcd(low, high)
    kwargs = {'log': True}
    if default:
        kwargs.update({'default_value': default})
    return UniformFloatHyperparameter(label, low, high, q=q, **kwargs)


def loguniform(label: str, low: float, high: float, default=None):
    kwargs = {'log': True}
    if default:
        kwargs.update({'default_value': default})
    return UniformFloatHyperparameter(label, low, high, **kwargs)


def int_qloguniform(label: str, low: int, high: int, q: int = None, default=None):
    if not q:
        q = min(low, 1)
    kwargs = {'log': True}
    if default:
        kwargs.update({'default_value': default})
    return UniformIntegerHyperparameter(label, low, high, q=q, **kwargs)


def int_loguniform(label: str, low: int, high: int, default=None):
    kwargs = {'log': True}
    if default:
        kwargs.update({'default_value': default})
    return UniformIntegerHyperparameter(label, low, high, **kwargs)
