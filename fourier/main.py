from Fourier import Fourier
from Drawing import Drawing
import make_gifs as mg


def main():

    d = Drawing()

    four = Fourier(d.xs, d.ys, iters=200)
    four.get_fourier_coefs()
    four.generate_new_line()

    xrange = (0, d.canvas_width)
    yrange = (-1*d.canvas_height, 0)

    mg.save_fourier_gif(four, xrange=xrange, yrange=yrange)

    return


if __name__ == '__main__':
    main()
