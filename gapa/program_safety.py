"""AST safety checks for LLM-generated GAPA play_once programs."""

from __future__ import annotations

import ast
from dataclasses import dataclass


ALLOWED_API_METHODS = {
    "pose",
    "choose_arm",
    "grasp",
    "move_up",
    "place_on",
    "place_in",
    "back_to_origin",
}


class ProgramSafetyError(ValueError):
    pass


@dataclass(frozen=True)
class SafetyReport:
    ok: bool
    allowed_api_methods: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "allowed_api_methods": list(self.allowed_api_methods),
        }


class _SafetyValidator(ast.NodeVisitor):
    def __init__(self):
        self.locals = {"api"}

    def fail(self, node: ast.AST, message: str) -> None:
        raise ProgramSafetyError(f"{message} at line {getattr(node, 'lineno', '?')}.")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name != "play_once":
            self.fail(node, "Only play_once(api) may be defined")
        if node.decorator_list:
            self.fail(node, "Decorators are not allowed")
        if node.returns is not None:
            self.fail(node, "Return annotations are not allowed")
        if node.args.vararg or node.args.kwarg or node.args.kwonlyargs or node.args.defaults:
            self.fail(node, "play_once must only accept api")
        if len(node.args.args) != 1 or node.args.args[0].arg != "api":
            self.fail(node, "play_once must have exactly one parameter named api")
        for stmt in node.body:
            self.visit(stmt)

    def visit_Import(self, node: ast.Import) -> None:
        self.fail(node, "Imports are not allowed")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.fail(node, "Imports are not allowed")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.fail(node, "Classes are not allowed")

    def visit_For(self, node: ast.For) -> None:
        self.fail(node, "Loops are not allowed")

    def visit_While(self, node: ast.While) -> None:
        self.fail(node, "Loops are not allowed")

    def visit_If(self, node: ast.If) -> None:
        self.fail(node, "Conditionals are not allowed")

    def visit_Try(self, node: ast.Try) -> None:
        self.fail(node, "Exception handling is not allowed")

    def visit_With(self, node: ast.With) -> None:
        self.fail(node, "Context managers are not allowed")

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.fail(node, "Lambdas are not allowed")

    def visit_Delete(self, node: ast.Delete) -> None:
        self.fail(node, "Delete statements are not allowed")

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None and not (isinstance(node.value, ast.Constant) and node.value.value is None):
            self.fail(node, "Only return None is allowed")

    def visit_Pass(self, node: ast.Pass) -> None:
        return

    def visit_Expr(self, node: ast.Expr) -> None:
        self.visit(node.value)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if not isinstance(target, ast.Name):
                self.fail(target, "Only simple local assignments are allowed")
            if target.id == "api":
                self.fail(target, "api cannot be reassigned")
            self.locals.add(target.id)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.fail(node, "Annotated assignments are not allowed")

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.fail(node, "Augmented assignments are not allowed")

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Attribute) or not isinstance(node.func.value, ast.Name):
            self.fail(node, "Only api.<skill>(...) calls are allowed")
        if node.func.value.id != "api":
            self.fail(node, "Only api.<skill>(...) calls are allowed")
        if node.func.attr not in ALLOWED_API_METHODS:
            self.fail(node, f"api.{node.func.attr} is not an allowed skill")
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            if keyword.arg is None:
                self.fail(keyword, "Expanded keyword arguments are not allowed")
            self.visit(keyword.value)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.fail(node, "Attribute access is only allowed as api.<skill>(...)")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.locals:
            self.fail(node, f"Unknown variable {node.id!r}")

    def visit_Constant(self, node: ast.Constant) -> None:
        return

    def visit_List(self, node: ast.List) -> None:
        for item in node.elts:
            self.visit(item)

    def visit_Tuple(self, node: ast.Tuple) -> None:
        for item in node.elts:
            self.visit(item)

    def visit_Dict(self, node: ast.Dict) -> None:
        for key, value in zip(node.keys, node.values):
            if key is not None:
                self.visit(key)
            self.visit(value)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, (ast.USub, ast.UAdd)):
            self.fail(node, "Only numeric unary operators are allowed")
        self.visit(node.operand)

    def generic_visit(self, node: ast.AST) -> None:
        self.fail(node, f"{node.__class__.__name__} is not allowed")


def validate_program_source(source: str) -> SafetyReport:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ProgramSafetyError(f"Program syntax error: {exc}") from exc

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(tree.body) != 1 or len(functions) != 1:
        raise ProgramSafetyError("Program must contain exactly one play_once(api) function.")

    validator = _SafetyValidator()
    validator.visit(functions[0])
    return SafetyReport(ok=True, allowed_api_methods=tuple(sorted(ALLOWED_API_METHODS)))
