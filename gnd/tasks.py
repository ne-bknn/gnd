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
        elif params is not None and not randomize:
            self.params = params
        else:
            # Ranomized not specified params

            new_params = self.randomize_params(**params)
            self.params = new_params

    def randomize_params(self, **kwargs) -> Dict[str, Any]:
        ...

    @staticmethod
    def get_param_boundaries() -> Dict[str, Dict[str, Any]]:
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

    def randomize_params(self, **kwargs) -> Dict[str, Any]:
        param_boundaries = self.get_param_boundaries()
        if "n" in kwargs:
            n = int(kwargs["n"])
        else:
            n = random.choice(param_boundaries["n"]["values"])
        assert isinstance(n, int)  # type narrowing
        if "type" in kwargs:
            t = kwargs["type"]
        else:
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

    def randomize_params(self, **kwargs) -> Dict[str, Any]:
        param_boundaries = self.get_param_boundaries()
        if "n" in kwargs:
            n = int(kwargs["n"])
        else:
            n = random.choice(param_boundaries["n"]["values"])

        if "m" in kwargs:
            m = int(kwargs["m"])
        else:
            m = random.choice(param_boundaries["m"]["values"])

        assert isinstance(n, int)  # type narrowing
        assert isinstance(m, int)  # type narrowing
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
            "m": m,
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
            "coordinates": {"values": ["physical", "algebraic"]},
        }

    def get_task(self) -> str:
        """Generates a task.

        Returns:
            str: Task
        """

        task = f"Преобразуй \\({{x(i)}}={sympy.latex(self.params['values'])}\\) матрицу с помощью преобразования {'Фурье' if self.params['type'] == 'fourier' else 'Хартли'}:\n"
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


class FourierByReverseFourier(AbstractTask):
    def __init__(
        self,
        params: Dict[str, str | int] | None = None,
        randomize: bool = False,
    ):
        super().__init__(params, randomize)

    @staticmethod
    def task_name():
        return "Обратное преобразование Фурье через прямое"

    @staticmethod
    def get_param_boundaries() -> Dict[str, Dict[str, Any]]:
        return {
            "n": {"values": [3, 4, 6, 8, 12]},
            "values": {"values": list(range(-5, 10))},
        }

    def randomize_params(self, **kwargs) -> Dict[str, Any]:
        param_boundaries = self.get_param_boundaries()
        if "n" in kwargs:
            n = int(kwargs["n"])
        else:
            n = random.choice(param_boundaries["n"]["values"])
        assert isinstance(n, int)
        v = sympy.Matrix(
            [random.choice(param_boundaries["values"]["values"]) for _ in range(n)]
        )

        params = {
            "n": n,
            "values": v,
        }

        return params

    def get_task(self):
        values = self.params["values"]
        matrix = sympy.latex(values)
        task = f"Примени к \\({{x(i)}}={matrix}\\) обратное преобразование Фурье через прямое:\n"
        return task

    @staticmethod
    def generate_matrix_fourier(N: int) -> sympy.Matrix:
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

    def get_solution(self):
        matrix = self.generate_matrix_fourier(self.params["n"])

        matrix_latex = sympy.latex(matrix)
        step1 = f"Матрица преобразования выглядит вот так:\n\\({matrix_latex}\\)"
        values = self.params["values"]
        result = matrix * values
        result_latex = sympy.latex(result)
        simplified = sympy.simplify(result)
        step2 = f"Перемножаем последовательность и матрицу:\n\\( X'(i) = {result_latex} = {sympy.latex(simplified)}\\)"
        step3 = f"Вторым шагом мы находим сопряженное к результату:\n\\({sympy.latex(simplified.H)}\\)"
        conjugate = simplified.H
        step4 = (
            "Делим на количество коэффициентов:\n"
            + "\\( X(i) = \\frac"
            + "{"
            + sympy.latex(conjugate)
            + "}{N} = \\frac"
            + "{"
            + sympy.latex(conjugate)
            + "}"
            + "{"
            + str(len(values))
            + "}"
            + "\\)"
        )

        print(step4)
        print(type(step4))

        return [
            ("Матрица преобразования", step1),
            ("Перемножение", step2),
            ("Третий шаг задачи", step3),
            ("Финальный шаг задачи", step4),
        ]


class HarmonicAmplitudeTask(AbstractTask):
    def __init__(
        self, params: Dict[str, str | int] | None = None, randomize: bool = False
    ):
        super().__init__(params, randomize)

    @staticmethod
    def task_name():
        return "Амплитуда N-ной гармоники"

    @staticmethod
    def get_param_boundaries() -> Dict[str, Dict[str, Any]]:
        return {
            "n": {"values": [3, 4, 6, 8, 12]},
            "values": {"values": list(range(-5, 10))},
            "i": {"values": list(range(1, 10))},
        }

    @staticmethod
    def generate_matrix_fourier(N: int) -> sympy.Matrix:
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

    def randomize_params(self, **kwargs) -> Dict[str, Any]:
        param_boundaries = self.get_param_boundaries()
        if "n" in kwargs:
            n = int(kwargs["n"])
        else:
            n = random.choice(param_boundaries["n"]["values"])
        assert isinstance(n, int)

        i = None
        if "i" in kwargs:
            i = int(kwargs["i"])
        else:
            i = random.choice(list(range(1, n // 2 + 1)))

        v = sympy.Matrix(
            [random.choice(param_boundaries["values"]["values"]) for _ in range(n)]
        )

        params = {
            "n": n,
            "values": v,
            "i": i,
        }

        return params

    def get_task(self):
        values = self.params["values"]
        matrix = sympy.latex(values)
        i = self.params["i"]
        task = f"Найди {i}-ую гармонику для \\({{x(i)}}={matrix}\\)\n"
        return task

    def get_solution(self):
        matrix = self.generate_matrix_fourier(self.params["n"])
        i = self.params["i"]
        n = self.params["n"]
        matrix_latex = sympy.latex(matrix)
        step1 = f"Матрица преобразования выглядит вот так:\n\\({matrix_latex}\\)"
        values = self.params["values"]
        result = matrix * values
        result_latex = sympy.latex(result)
        simplified = sympy.simplify(result)
        step2 = f"Перемножаем последовательность и матрицу:\n\\( X(i) = {result_latex} = {sympy.latex(simplified)}\\)"
        step3 = (
            f"Возьмем модуль i-ого элемента из преобразованной последовательности, поделим на количество элементов и получим:\n"
            + "\\(\\frac{|"
            + sympy.latex(simplified[i])
            + "|}{n} = \\frac{"
            + sympy.latex(abs(simplified[i]))
            + "}{"
            + str(n)
            + "}\\)"
        )

        return [
            ("Матрица преобразования", step1),
            ("Перемножение", step2),
            ("Последний шаг", step3),
        ]


TaskMap = {
    "transform_1d": FourierHartleyTask,
    "transoform_2d": FourierHartley2dTask,
    "reverse": FourierByReverseFourier,
    "harmonic_amplitude": HarmonicAmplitudeTask,
}
