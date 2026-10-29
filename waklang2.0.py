import ast
import operator
import re

from kivy.app import App
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup

Window.clearcolor = (0.055, 0.065, 0.10, 1)


class WakLangError(Exception):
    pass


class ReturnValue(Exception):
    def __init__(self, value):
        self.value = value


class BreakLoop(Exception):
    pass


class ContinueLoop(Exception):
    pass


class WakLang:
    def __init__(self, input_callback=None):
        self.variables = {}
        self.functions = {}
        self.output = []
        self.input_callback = input_callback
        self.steps = 0

    # ---------- EXPRESSIONS ----------

    def evaluate(self, expression):
        expression = expression.strip()

        if not expression:
            raise WakLangError("An expression is missing.")

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            raise WakLangError(
                f"Invalid expression: {expression}"
            )

        binary_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }

        comparisons = {
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.In: lambda a, b: a in b,
            ast.NotIn: lambda a, b: a not in b,
        }

        def visit(node):
            if isinstance(node, ast.Expression):
                return visit(node.body)

            if isinstance(node, ast.Constant):
                if type(node.value) in (
                    str, int, float, bool, type(None)
                ):
                    return node.value

                raise WakLangError("Unsupported constant.")

            if isinstance(node, ast.Name):
                if node.id == "true":
                    return True
                if node.id == "false":
                    return False
                if node.id == "null":
                    return None

                if node.id in self.variables:
                    return self.variables[node.id]

                raise WakLangError(
                    f"Unknown variable: {node.id}"
                )

            if isinstance(node, ast.List):
                return [visit(item) for item in node.elts]

            if isinstance(node, ast.Tuple):
                return tuple(visit(item) for item in node.elts)

            if isinstance(node, ast.Dict):
                return {
                    visit(k): visit(v)
                    for k, v in zip(node.keys, node.values)
                }

            if isinstance(node, ast.BinOp):
                operation = binary_ops.get(type(node.op))

                if operation is None:
                    raise WakLangError(
                        "Unsupported arithmetic operator."
                    )

                # Avoid excessively large exponent calculations.
                if isinstance(node.op, ast.Pow):
                    exponent = visit(node.right)

                    if (
                        not isinstance(exponent, (int, float))
                        or abs(exponent) > 1000
                    ):
                        raise WakLangError(
                            "Exponent is outside the allowed range."
                        )

                    left = visit(node.left)

                    try:
                        return operator.pow(left, exponent)
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise WakLangError(str(exc))

                try:
                    return operation(
                        visit(node.left),
                        visit(node.right)
                    )
                except (
                    TypeError, ValueError, ZeroDivisionError,
                    OverflowError
                ) as exc:
                    raise WakLangError(str(exc))

            if isinstance(node, ast.UnaryOp):
                value = visit(node.operand)

                if isinstance(node.op, ast.USub):
                    return -value
                if isinstance(node.op, ast.UAdd):
                    return +value
                if isinstance(node.op, ast.Not):
                    return not value

                raise WakLangError("Unsupported unary operator.")

            if isinstance(node, ast.BoolOp):
                if isinstance(node.op, ast.And):
                    for item in node.values:
                        result = visit(item)
                        if not result:
                            return result
                    return result

                if isinstance(node.op, ast.Or):
                    for item in node.values:
                        result = visit(item)
                        if result:
                            return result
                    return result

            if isinstance(node, ast.Compare):
                left = visit(node.left)

                for op_node, right_node in zip(
                    node.ops, node.comparators
                ):
                    operation = comparisons.get(type(op_node))

                    if operation is None:
                        raise WakLangError(
                            "Unsupported comparison."
                        )

                    right = visit(right_node)

                    try:
                        if not operation(left, right):
                            return False
                    except (TypeError, ValueError) as exc:
                        raise WakLangError(str(exc))

                    left = right

                return True

            if isinstance(node, ast.IfExp):
                if visit(node.test):
                    return visit(node.body)
                return visit(node.orelse)

            if isinstance(node, ast.Subscript):
                collection = visit(node.value)
                index = visit(node.slice)

                try:
                    return collection[index]
                except (IndexError, KeyError, TypeError) as exc:
                    raise WakLangError(str(exc))

            if isinstance(node, ast.Slice):
                lower = visit(node.lower) if node.lower else None
                upper = visit(node.upper) if node.upper else None
                step = visit(node.step) if node.step else None
                return slice(lower, upper, step)

            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise WakLangError(
                        "Only named functions can be called."
                    )

                name = node.func.id
                args = [visit(arg) for arg in node.args]

                builtins = {
                    "length": lambda x: len(x),
                    "str": lambda x: str(x),
                    "int": self.to_int,
                    "float": self.to_float,
                    "type": lambda x: self.type_name(x),
                    "abs": lambda x: abs(x),
                    "round": lambda x: round(x),
                    "min": lambda *x: min(*x),
                    "max": lambda *x: max(*x),
                    "sum": lambda x: sum(x),
                }

                if name in builtins:
                    try:
                        return builtins[name](*args)
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise WakLangError(str(exc))

                if name in self.functions:
                    return self.call_function(name, args)

                raise WakLangError(
                    f"Unknown function: {name}"
                )

            raise WakLangError(
                f"Unsupported expression type: "
                f"{type(node).__name__}"
            )

        return visit(tree)

    @staticmethod
    def to_int(value):
        return int(value)

    @staticmethod
    def to_float(value):
        return float(value)

    @staticmethod
    def type_name(value):
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, str):
            return "string"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "float"
        if isinstance(value, list):
            return "list"
        if value is None:
            return "null"
        return "object"

    # ---------- PARSER ----------

    def read_block(self, lines, start):
        body = []
        depth = 1

        while start < len(lines):
            line = lines[start].strip()
            start += 1

            # Ignore braces inside quoted strings.
            cleaned = self.remove_strings(line)

            if cleaned.rstrip().endswith("{"):
                depth += 1

            if cleaned.strip() == "}":
                depth -= 1

                if depth == 0:
                    return body, start

            body.append(line)

        raise WakLangError("Missing closing brace '}'.")

    @staticmethod
    def remove_strings(line):
        return re.sub(
            r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
            '""',
            line
        )

    # ---------- FUNCTIONS ----------

    def call_function(self, name, args):
        parameters, body = self.functions[name]

        if len(args) != len(parameters):
            raise WakLangError(
                f"{name} expects {len(parameters)} argument(s), "
                f"but received {len(args)}."
            )

        old_variables = self.variables.copy()
        self.variables = old_variables.copy()

        for parameter, value in zip(parameters, args):
            self.variables[parameter] = value

        try:
            self.execute_block(body)
        except ReturnValue as result:
            return result.value
        finally:
            self.variables = old_variables

        return None

    # ---------- STATEMENTS ----------

    def execute_block(self, lines, depth=0):
        if depth > 50:
            raise WakLangError(
                "Maximum nesting or function depth exceeded."
            )

        i = 0

        while i < len(lines):
            self.steps += 1

            if self.steps > 100000:
                raise WakLangError(
                    "Execution limit reached. Check your loops."
                )

            line = lines[i].strip()
            line_number = i + 1
            i += 1

            if not line or line.startswith("#"):
                continue

            try:
                # PRINT / SHOW
                if line.startswith("show "):
                    self.output.append(
                        str(self.evaluate(line[5:]))
                    )
                    continue

                # IF / ELSE
                if line.startswith("if ") and line.endswith("{"):
                    condition = self.evaluate(line[3:-1])

                    true_body, i = self.read_block(lines, i)
                    false_body = []

                    if (
                        i < len(lines)
                        and lines[i].strip() == "else {"
                    ):
                        false_body, i = self.read_block(
                            lines, i + 1
                        )

                    if condition:
                        self.execute_block(true_body, depth + 1)
                    else:
                        self.execute_block(false_body, depth + 1)

                    continue

                # REPEAT
                if line.startswith("repeat ") and line.endswith("{"):
                    count = self.evaluate(line[7:-1])

                    if (
                        not isinstance(count, (int, float))
                        or isinstance(count, bool)
                    ):
                        raise WakLangError(
                            "Repeat count must be a number."
                        )

                    count = int(count)

                    if count < 0 or count > 10000:
                        raise WakLangError(
                            "Repeat count must be between 0 and 10000."
                        )

                    body, i = self.read_block(lines, i)

                    for _ in range(count):
                        try:
                            self.execute_block(body, depth + 1)
                        except ContinueLoop:
                            continue
                        except BreakLoop:
                            break

                    continue

                # WHILE
                if line.startswith("while ") and line.endswith("{"):
                    condition = line[6:-1].strip()
                    body, i = self.read_block(lines, i)

                    while self.evaluate(condition):
                        try:
                            self.execute_block(body, depth + 1)
                        except ContinueLoop:
                            continue
                        except BreakLoop:
                            break

                    continue

                # FUNCTION DECLARATION
                if line.startswith("function ") and line.endswith("{"):
                    declaration = line[9:-1].strip()

                    match = re.fullmatch(
                        r"([A-Za-z_]\w*)\s*"
                        r"\((.*?)\)",
                        declaration
                    )

                    if not match:
                        raise WakLangError(
                            "Use: function name(parameter) {"
                        )

                    name, parameter_text = match.groups()

                    parameters = (
                        []
                        if not parameter_text.strip()
                        else [
                            p.strip()
                            for p in parameter_text.split(",")
                        ]
                    )

                    if not all(
                        re.fullmatch(r"[A-Za-z_]\w*", p)
                        for p in parameters
                    ):
                        raise WakLangError(
                            "Invalid function parameter."
                        )

                    if len(set(parameters)) != len(parameters):
                        raise WakLangError(
                            "Duplicate function parameter."
                        )

                    body, i = self.read_block(lines, i)
                    self.functions[name] = (parameters, body)
                    continue

                # RETURN
                if line == "return":
                    raise ReturnValue(None)

                if line.startswith("return "):
                    raise ReturnValue(
                        self.evaluate(line[7:])
                    )

                # BREAK / CONTINUE
                if line == "break":
                    raise BreakLoop()

                if line == "continue":
                    raise ContinueLoop()

                # INPUT
                if line.startswith("input "):
                    name = line[6:].strip()

                    if not re.fullmatch(r"[A-Za-z_]\w*", name):
                        raise WakLangError(
                            "Use: input variable_name"
                        )

                    if self.input_callback is None:
                        raise WakLangError(
                            "Input is not connected to the editor yet."
                        )

                    self.variables[name] = self.input_callback(name)
                    continue

                # LIST ITEM ASSIGNMENT
                match = re.fullmatch(
                    r"([A-Za-z_]\w*)\[(.+)\]\s*=\s*(.+)",
                    line
                )

                if match:
                    name, index_expr, value_expr = match.groups()

                    if name not in self.variables:
                        raise WakLangError(
                            f"Unknown list: {name}"
                        )

                    try:
                        self.variables[name][
                            self.evaluate(index_expr)
                        ] = self.evaluate(value_expr)
                    except (IndexError, KeyError, TypeError) as exc:
                        raise WakLangError(str(exc))

                    continue

                # VARIABLE ASSIGNMENT
                match = re.fullmatch(
                    r"([A-Za-z_]\w*)\s*=\s*(.+)",
                    line
                )

                if match:
                    name, expression = match.groups()

                    reserved = {
                        "show", "if", "else", "repeat",
                        "while", "function", "return",
                        "input", "break", "continue"
                    }

                    if name in reserved:
                        raise WakLangError(
                            f"Reserved word: {name}"
                        )

                    self.variables[name] = self.evaluate(expression)
                    continue

                raise WakLangError(
                    "Unknown command: " + line
                )

            except (WakLangError, ReturnValue, BreakLoop, ContinueLoop):
                raise
            except Exception as exc:
                raise WakLangError(
                    f"Statement error: {exc}"
                )

    # ---------- PROGRAM ENTRY ----------

    def run(self, source):
        self.variables = {}
        self.functions = {}
        self.output = []
        self.steps = 0

        lines = source.splitlines()

        try:
            self.execute_block(lines)
        except WakLangError as exc:
            raise WakLangError(str(exc))
        except ReturnValue:
            raise WakLangError(
                "'return' can only be used inside a function."
            )
        except BreakLoop:
            raise WakLangError(
                "'break' can only be used inside a loop."
            )
        except ContinueLoop:
            raise WakLangError(
                "'continue' can only be used inside a loop."
            )

        return "\n".join(self.output)


# ==================================================
# WAKLANG GRAPHICAL EDITOR
# ==================================================

class WakLangEditor(App):
    def build(self):
        self.title = "WakLang 2.0"

        root = BoxLayout(
            orientation="vertical",
            padding=dp(10),
            spacing=dp(7)
        )

        heading = Label(
            text="[b]WAKLANG[/b] [color=818CF8]2.0[/color]",
            markup=True,
            font_size="23sp",
            size_hint_y=None,
            height=dp(42)
        )
        root.add_widget(heading)

        self.editor = TextInput(
            text=(
                '# Welcome to WakLang 2.0\n'
                'name = "WakLang"\n'
                'age = 15\n'
                '\n'
                'show "Hello from my language!"\n'
                'show name\n'
                '\n'
                'if age >= 18 {\n'
                '    show "Adult"\n'
                '}\n'
                'else {\n'
                '    show "Not an adult"\n'
                '}\n'
                '\n'
                'repeat 3 {\n'
                '    show "WakLang is growing!"\n'
                '}\n'
            ),
            multiline=True,
            font_size="15sp",
            background_color=(0.09, 0.11, 0.17, 1),
            foreground_color=(0.88, 0.91, 1, 1),
            cursor_color=(0.5, 0.6, 1, 1),
            size_hint_y=0.62
        )
        root.add_widget(self.editor)

        buttons = BoxLayout(
            size_hint_y=None,
            height=dp(45),
            spacing=dp(6)
        )

        run_button = Button(
            text="▶ RUN",
            background_normal="",
            background_color=(0.25, 0.32, 0.85, 1)
        )
        run_button.bind(on_press=self.run_program)
        buttons.add_widget(run_button)

        example_button = Button(
            text="EXAMPLE",
            background_normal="",
            background_color=(0.12, 0.48, 0.40, 1)
        )
        example_button.bind(on_press=self.load_example)
        buttons.add_widget(example_button)

        clear_button = Button(
            text="CLEAR",
            background_normal="",
            background_color=(0.32, 0.33, 0.40, 1)
        )
        clear_button.bind(on_press=self.clear_output)
        buttons.add_widget(clear_button)

        root.add_widget(buttons)

        root.add_widget(Label(
            text="[b]OUTPUT CONSOLE[/b]",
            markup=True,
            size_hint_y=None,
            height=dp(28)
        ))

        self.output = TextInput(
            text="WakLang 2.0 is ready.",
            readonly=True,
            multiline=True,
            font_size="14sp",
            background_color=(0.025, 0.035, 0.06, 1),
            foreground_color=(0.45, 0.95, 0.70, 1),
            size_hint_y=0.28
        )
        root.add_widget(self.output)

        return root

    def run_program(self, button):
        try:
            interpreter = WakLang()
            result = interpreter.run(self.editor.text)

            self.output.text = (
                result if result
                else "Program finished successfully."
            )

        except Exception as exc:
            self.output.text = (
                "WakLang Error:\n" + str(exc)
            )

    def clear_output(self, button):
        self.output.text = ""

    def load_example(self, button):
        self.editor.text = (
            '# WakLang example\n'
            'function add(a, b) {\n'
            '    return a + b\n'
            '}\n'
            '\n'
            'answer = add(10, 20)\n'
            'show answer\n'
            '\n'
            'scores = [80, 90, 95]\n'
            'show scores[0]\n'
            'show length(scores)\n'
            '\n'
            'age = 15\n'
            'if age >= 18 {\n'
            '    show "Adult"\n'
            '}\n'
            'else {\n'
            '    show "Not an adult"\n'
            '}\n'
            '\n'
            'repeat 3 {\n'
            '    show "Hello!"\n'
            '}\n'
        )


if __name__ == "__main__":
    WakLangEditor().run()