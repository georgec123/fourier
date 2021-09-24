import numpy as np


class Fourier:
    """
    Generate fourier coefficients for a given 2d curve
    """
    def __init__(self, x: np.array, y: np.array, iters: int = 200):

        self.x = x
        self.y = y

        assert len(x) == len(y)

        self.compl = x + y * 1j

        self.line_len = len(x)
        self.dt = 1 / (1 + len(x))

        self.ts = np.linspace(0, 1, num=self.line_len)

        self.iter_list = np.linspace(-1 * iters, iters, 2 * iters + 1)

        self.coefs = None
        self.gen_line = None

        return

    @staticmethod
    def get_fourier_coef(f: np.array, ts: np.array, dt: float, n: int = 0) -> np.array:
        """
        Get n'th fourier coefficient via integral method.
        Uses sum approximation

        :param f: Complex array representing 2d curve
        :param ts: Parametric variable array
        :param dt: Spacing between each element in ts
        :param n: nth coefficient
        :return:
        """
        coef = np.sum(f[1:] * dt * np.exp(-2 * np.pi * n * 1j * ts[1:]))

        return np.asarray(coef)

    def get_fourier_coefs(self) -> np.array:
        """
        Get list of n fourier coefficients
        :return: list of fourier coefficients
        """

        fourier_coef_list = [self.get_fourier_coef(self.compl, self.ts, self.dt, n) for n in self.iter_list]

        return np.asarray(fourier_coef_list)

    def generate_new_line(self) -> None:
        """
        Generate approximate curve given fourier coefficients
        """
        if self.coefs is None:
            self.coefs = self.get_fourier_coefs()

        zs = []

        for t in self.ts:

            z = np.sum([coef_n * np.exp(2 * np.pi * n * 1j * t) for coef_n, n in zip(self.coefs, self.iter_list)])
            zs.append(z)

        line = np.asarray(zs)

        # known bug, need to shuffle line as plot looks strange otherwise
        shift_line = np.append(line[1:], line[0])[1:-1]

        self.gen_line = shift_line

        return
