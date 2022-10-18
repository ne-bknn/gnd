import sympy
import random
from typing import Any, Dict, List


class AbstractTask:
    def __init__(
        self,
        params: Dict[str, str | int] | None = None,
        randomize: bool = False,
    ):
        self.params: Dict[str, str | int]
        if params is None and randomize:
            self.params = self.randomize_params()
        elif params is not None:
            self.params = params

    def randomize_params(self) -> Dict[str, Any]:
        ...

    def get_param_boundaries(self) -> Dict[str, Dict[str, Any]]:
        ...

    def get_task(self) -> str:
        ...

    def get_solution(self) -> list[str]:
        ...


class FourierHartleyTask(AbstractTask):
    def __init__(
        self,
        params: Dict[str, str | int] | None = None,
        randomize: bool = False,
    ):
        super().__init__(params, randomize)

    @staticmethod
    def task_name() -> str:
        return "Одномерное ДПФ/ДПХ"

    def randomize_params(self) -> Dict[str, Any]:
        param_boundaries = self.get_param_boundaries()
        n = random.choice(param_boundaries["n"]["values"])
        assert isinstance(n, int)  # type narrowing
        t = random.choice(param_boundaries["type"]["values"])
        assert isinstance(t, str)  # type narrowing
        v = [random.choice(param_boundaries["values"]["values"]) for _ in range(n)]
        params = {
            "n": n,
            "type": t,
            "values": v,
        }
        return params

    @staticmethod
    def get_param_boundaries() -> Dict[str, Dict[str, Any]]:
        return {
            "n": {"values": [3, 4, 6, 8, 12]},
            "type": {"values": ["fourier", "hartley"]},
            "values": {"values": list(range(-5, 10))},
        }

    def get_task(self) -> str:
        """Generates a task.

        Returns:
            str: Task
        """

        task = f"Преобразуй \\({{x(i)}}={self.params['values']}\\) последовательность с помощью преобразования {'Фурье' if self.params['type'] == 'fourier' else 'Хартли'}:\n"
        return task

    def get_solution(self) -> list[str]:
        """Generates a step-by-step solution for the task.

        Returns:
            list[str]: List of steps for the solution
        """
        if self.params["type"] == "fourier":
            return self.get_fourier_solution()
        else:
            return self.get_hartley_solution()

    @staticmethod
    def generate_matrix_fourier(N: int) -> sympy.Matrix:
        """Generates a matrix for Fourier transform.

        Args:
            n (int): Size of the matrix

        Returns:
            sympy.Matrix: Generated matrix
        """
        frac = sympy.Rational(2, N)
        matrix: List[List[sympy.Expr]] = []
        for i in range(N):
            matrix.append([])
            for j in range(N):
                matrix[i].append(
                    sympy.simplify(
                        sympy.cos(frac * sympy.pi * i * j)
                        - 1j * sympy.sin(frac * sympy.pi * i * j)
                    )
                )

        return sympy.Matrix(matrix)

    @staticmethod
    def generate_matrix_hartley(N: int):
        frac = sympy.Rational(2, N)
        matrix: list[list[sympy.Expr]] = []
        for i in range(N):
            matrix.append([])
            for j in range(N):
                matrix[i].append(
                    sympy.simplify(
                        sympy.cos(frac * sympy.pi * i * j)
                        + sympy.sin(frac * sympy.pi * i * j)
                    )
                )

        return sympy.Matrix(matrix)

    def get_fourier_solution(self) -> list[tuple[str, str]]:
        matrix = self.generate_matrix_fourier(self.params["n"])

        matrix_latex = sympy.latex(matrix)
        step1 = f"Матрица преобразования выглядит вот так:\n\\({matrix_latex}\\)"
        result = sympy.latex(matrix * sympy.Matrix(self.params["values"]))
        step2 = f"Результат преобразования:\n\\( X(i) = {result}\\)"
        return [("Матрица преобразования", step1), ("Результат", step2)]

    def get_hartley_solution(self) -> list[tuple[str, str]]:
        matrix = self.generate_matrix_hartley(self.params["n"])

        matrix_latex = sympy.latex(matrix)
        step1 = f"Матрица преобразования выглядит вот так:\n\\({matrix_latex}\\)"
        result = sympy.latex(matrix * sympy.Matrix(self.params["values"]))
        step2 = f"Результат преобразования:\n\\( H(i) = {result}\\)"
        return [("Матрица преобразования", step1), ("Результат", step2)]


class FourierHartley2dTask(AbstractTask):
    def __init__(
        self,
        params: Dict[str, str | int] | None = None,
        randomize: bool = False,
    ):
        super().__init__(params, randomize)

    @staticmethod
    def task_name() -> str:
        return "Двумерное ДПФ/ДПХ"

    def randomize_params(self) -> Dict[str, Any]:
        param_boundaries = self.get_param_boundaries()
        n = random.choice(param_boundaries["n"]["values"])
        m = random.choice(param_boundaries["m"]["values"])
        assert isinstance(n, int)  # type narrowing
        t = random.choice(param_boundaries["type"]["values"])
        assert isinstance(t, str)  # type narrowing
        v = sympy.Matrix(
            [
                [random.choice(param_boundaries["values"]["values"]) for _ in range(n)]
                for _ in range(m)
            ]
        )
        params = {
            "n": n,
            "type": t,
            "values": v,
        }
        return params

    @staticmethod
    def get_param_boundaries() -> Dict[str, Dict[str, Any]]:
        return {
            "n": {"values": [3, 4, 6, 8, 12]},
            "m": {"values": [3, 4, 6, 8, 12]},
            "type": {"values": ["fourier", "hartley"]},
            "values": {"values": list(range(-5, 10))},
        }

    def get_task(self) -> str:
        """Generates a task.

        Returns:
            str: Task
        """

        task = f"Преобразуй \\({{x(i)}}={sympy.latex(self.params['values'])}\\) последовательность с помощью преобразования {'Фурье' if self.params['type'] == 'fourier' else 'Хартли'}:\n"
        return task

    def get_solution(self) -> list[str]:
        """Generates a step-by-step solution for the task.

        Returns:
            list[str]: List of steps for the solution
        """
        if self.params["type"] == "fourier":
            return self.get_fourier_solution()
        else:
            return self.get_hartley_solution()

    @staticmethod
    def generate_matrix_fourier(N: int) -> sympy.Matrix:
        """Generates a matrix for Fourier transform.

        Args:
            n (int): Size of the matrix

        Returns:
            sympy.Matrix: Generated matrix
        """
        frac = sympy.Rational(2, N)
        matrix: List[List[sympy.Expr]] = []
        for i in range(N):
            matrix.append([])
            for j in range(N):
                matrix[i].append(
                    sympy.simplify(
                        sympy.cos(frac * sympy.pi * i * j)
                        - 1j * sympy.sin(frac * sympy.pi * i * j)
                    )
                )

        return sympy.Matrix(matrix)

    @staticmethod
    def generate_matrix_hartley(N: int):
        frac = sympy.Rational(2, N)
        matrix: list[list[sympy.Expr]] = []
        for i in range(N):
            matrix.append([])
            for j in range(N):
                matrix[i].append(
                    sympy.simplify(
                        sympy.cos(frac * sympy.pi * i * j)
                        + sympy.sin(frac * sympy.pi * i * j)
                    )
                )

        return sympy.Matrix(matrix)

    def get_fourier_solution(self) -> list[tuple[str, str]]:
        matrix = self.generate_matrix_fourier(self.params["n"])

        matrix_latex = sympy.latex(matrix)
        step1 = f"Матрица преобразования выглядит вот так:\n\\({matrix_latex}\\)"
        result = sympy.latex(matrix * sympy.Matrix(self.params["values"]))
        step2 = f"Результат преобразования:\n\\( X(i) = {result}\\)"
        return [("Матрица преобразования", step1), ("Результат", step2)]

    def get_hartley_solution(self) -> list[tuple[str, str]]:
        matrix = self.generate_matrix_hartley(self.params["n"])

        matrix_latex = sympy.latex(matrix)
        step1 = f"Матрица преобразования выглядит вот так:\n\\({matrix_latex}\\)"
        result = sympy.latex(matrix * sympy.Matrix(self.params["values"]))
        step2 = f"Результат преобразования:\n\\( H(i) = {result}\\)"
        return [("Матрица преобразования", step1), ("Результат", step2)]


TaskMap = {"transform_1d": FourierHartleyTask, "transoform_2d": FourierHartley2dTask}
