import random
from typing import Any, Dict


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
            "type": {"values": ["Fourier", "Hartley"]},
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
        if self.params["type"] == "Fourier":
            return self.get_fourier_solution()
        else:
            return self.get_hartley_solution()

    def get_fourier_solution(self) -> list[str]:
        step1 = "Fourier transform looks like this:\n\\("
        return [step1]

    def get_hartley_solution(self) -> list[str]:
        step1 = "Fourier transform looks like this:\n\\("
        return [step1]


TaskMap = {"transform_1d": FourierHartleyTask}
