from __future__ import annotations

import ast
import json
import math
import operator
from typing import Any


MAX_CODE_CHARS = 8000
MAX_INPUT_BYTES = 50_000
MAX_STEPS = 20_000
MAX_OUTPUT_CHARS = 6000
MAX_COLLECTION = 1000

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda left, right: left in right,
    ast.NotIn: lambda left, right: left not in right,
}
_FUNCTIONS = {
    "abs": abs,
    "len": len,
    "max": max,
    "min": min,
    "round": round,
    "sorted": sorted,
    "sum": sum,
}


class SandboxError(ValueError):
    pass


class _Interpreter:
    def __init__(self, variables: dict[str, Any]) -> None:
        self.variables = variables
        self.output: list[str] = []
        self.steps = 0

    def step(self) -> None:
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise SandboxError("执行步数超过上限，已终止分析。")

    def value(self, node: ast.AST) -> Any:
        self.step()
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, str) and len(value) > 2000:
                raise SandboxError("字符串常量过长。")
            if isinstance(value, int) and value.bit_length() > 128:
                raise SandboxError("整数常量过大。")
            return value
        if isinstance(node, ast.Name):
            if node.id not in self.variables:
                raise SandboxError(f"变量未定义：{node.id}")
            return self.variables[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            if len(node.elts) > MAX_COLLECTION:
                raise SandboxError("集合元素超过上限。")
            values = [self.value(item) for item in node.elts]
            return values if isinstance(node, ast.List) else tuple(values)
        if isinstance(node, ast.Dict):
            if len(node.keys) > MAX_COLLECTION:
                raise SandboxError("字典元素超过上限。")
            return {self.value(key): self.value(value) for key, value in zip(node.keys, node.values)}
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left, right = self.value(node.left), self.value(node.right)
            if isinstance(node.op, ast.Pow) and (not isinstance(right, (int, float)) or abs(right) > 8):
                raise SandboxError("幂运算指数最多为 8。")
            if isinstance(node.op, ast.Mult):
                sequence, count = (left, right) if isinstance(right, int) else (right, left)
                if isinstance(sequence, (str, list, tuple)) and isinstance(count, int):
                    if len(sequence) * max(0, count) > MAX_COLLECTION:
                        raise SandboxError("运算结果集合过大。")
            return self._checked(_BINARY[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
            value = self.value(node.operand)
            if isinstance(node.op, ast.Not):
                return not value
            return self._checked(value if isinstance(node.op, ast.UAdd) else -value)
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            values = [self.value(item) for item in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare):
            left = self.value(node.left)
            for operation, comparator in zip(node.ops, node.comparators):
                if type(operation) not in _COMPARE:
                    raise SandboxError("不支持该比较操作。")
                right = self.value(comparator)
                if not _COMPARE[type(operation)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self.value(node.body if self.value(node.test) else node.orelse)
        if isinstance(node, ast.Subscript):
            container, index = self.value(node.value), self.value(node.slice)
            if not isinstance(container, (dict, list, tuple, str)):
                raise SandboxError("只允许读取字典、列表、元组和字符串。")
            return container[index]
        if isinstance(node, ast.Slice):
            return slice(
                self.value(node.lower) if node.lower else None,
                self.value(node.upper) if node.upper else None,
                self.value(node.step) if node.step else None,
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            args = [self.value(arg) for arg in node.args]
            if node.func.id == "range":
                if not 1 <= len(args) <= 3 or any(not isinstance(item, int) or abs(item) > 1_000_000 for item in args):
                    raise SandboxError("range 只接受 1 到 3 个整数参数。")
                result = range(*args)
                if len(result) > 2000:
                    raise SandboxError("单次循环最多 2000 步。")
                return result
            function = _FUNCTIONS.get(node.func.id)
            if function is None:
                raise SandboxError(f"不允许调用函数：{node.func.id}")
            return self._checked(function(*args))
        raise SandboxError(f"不支持的语法：{type(node).__name__}")

    @staticmethod
    def _checked(value: Any) -> Any:
        if isinstance(value, float) and (not math.isfinite(value) or abs(value) > 1e100):
            raise SandboxError("计算结果超出允许范围。")
        if isinstance(value, int) and value.bit_length() > 128:
            raise SandboxError("整数结果超出允许范围。")
        if isinstance(value, (str, list, tuple, dict)) and len(value) > MAX_COLLECTION:
            raise SandboxError("计算结果集合过大。")
        return value

    def statement(self, node: ast.stmt) -> None:
        self.step()
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.startswith("_") or name in _FUNCTIONS or name == "range":
                raise SandboxError("变量名无效或与内置函数冲突。")
            self.variables[name] = self.value(node.value)
            return
        if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name) and type(node.op) in _BINARY:
            name = node.target.id
            if name.startswith("_") or name not in self.variables:
                raise SandboxError("复合赋值目标必须是已定义的普通变量。")
            self.variables[name] = self._checked(_BINARY[type(node.op)](self.variables[name], self.value(node.value)))
            return
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id != "print" or node.value.keywords:
                raise SandboxError("表达式语句只允许 print(...)。")
            rendered = " ".join(str(self.value(arg)) for arg in node.value.args)
            self.output.append(rendered)
            if sum(map(len, self.output)) > MAX_OUTPUT_CHARS:
                raise SandboxError("输出超过长度上限。")
            return
        if isinstance(node, ast.If):
            branch = node.body if self.value(node.test) else node.orelse
            for statement in branch:
                self.statement(statement)
            return
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            name = node.target.id
            if name.startswith("_") or not isinstance(node.iter, ast.Call) or not isinstance(node.iter.func, ast.Name) or node.iter.func.id != "range":
                raise SandboxError("for 循环只允许遍历受限的 range(...)。")
            values = self.value(node.iter)
            for value in values:
                self.step()
                self.variables[name] = value
                for statement in node.body:
                    self.statement(statement)
            return
        raise SandboxError(f"不支持的语句：{type(node).__name__}")


def execute_analysis(code: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    if not code.strip():
        raise SandboxError("请填写要运行的分析代码。")
    if len(code) > MAX_CODE_CHARS:
        raise SandboxError(f"代码不能超过 {MAX_CODE_CHARS} 个字符。")
    variables = variables or {}
    if not isinstance(variables, dict) or "data" in variables:
        raise SandboxError("输入变量必须是 JSON 对象，且不能覆盖保留变量 data。")
    if any(not isinstance(key, str) or not key.isidentifier() or key.startswith("_") for key in variables):
        raise SandboxError("输入变量名必须是普通 Python 标识符，且不能以下划线开头。")
    try:
        encoded = json.dumps(variables, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise SandboxError("输入变量必须是有限数值、字符串、列表或对象。") from error
    if len(encoded) > MAX_INPUT_BYTES:
        raise SandboxError("输入数据不能超过 50 KB。")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as error:
        raise SandboxError(f"Python 语法错误（第 {error.lineno} 行）：{error.msg}") from error
    if len(list(ast.walk(tree))) > 800:
        raise SandboxError("代码结构过于复杂。")
    interpreter = _Interpreter({**variables})
    try:
        for statement in tree.body:
            interpreter.statement(statement)
    except SandboxError:
        raise
    except (ArithmeticError, TypeError, ValueError, KeyError, IndexError) as error:
        raise SandboxError(f"分析运行失败：{error}") from error
    return {
        "status": "completed",
        "output": "\n".join(interpreter.output),
        "variables": interpreter.variables,
        "steps": interpreter.steps,
        "sandbox": "restricted_python_ast_v1",
    }
