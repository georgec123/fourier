import os
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fourier import funcs
from fourier.Fourier import Fourier
import moviepy.editor as mp
from typing import Tuple
import pandas as pd


def animate(info: Tuple[int, pd.DataFrame, np.array], line1: Line2D, line2: Line2D) -> Tuple[Line2D, Line2D]:
    """
    Produce plot for each frame in the gif
    :param info: Tuple of (index, grouped fourier dataframe, fourier line)
    :param line1:
    :param line2:
    :return:
    """
    idx = info[0]
    df_group = info[1]
    gen_line = info[2]

    if idx % 50 == 0:
        print(f"{idx}/{len(gen_line)}")

    gen_line = gen_line[:idx]

    gen_x, gen_y = np.real(gen_line), np.imag(gen_line)

    line1.set_data(df_group['x_cum'], df_group['y_cum'])
    line2.set_data(gen_x, gen_y)

    return line1, line2


def save_fourier_gif(fourier: Fourier, save_path=r'C:\temp\myfirstAnimation.gif', xrange: Tuple[int, int] = (0, 500),
                     yrange: Tuple[int, int] = (-500, 0)):

    df = funcs.get_fourier_df_for_plotting(fourier=fourier)

    fig = plt.figure()
    ax = plt.axes(xlim=xrange, ylim=yrange)

    line1, = ax.plot([], [], lw=2)
    line2, = ax.plot([], [], lw=2)

    grouped_df = df.groupby(by='t')

    print(f"Beginning gif creation")
    ani = animation.FuncAnimation(fig, animate,
                                  frames=[(idx, d[1], fourier.gen_line) for idx, d in enumerate(grouped_df)],
                                  interval=1, blit=True, repeat=False, fargs=(line1, line2,))

    ani.save(save_path, fps=100000)
    print(f"Saved gif at {save_path}")

    return


def speed_up_gif(file_path: str, speed_mult: float = 2) -> None:
    """
    Speed up gif by a given multiplier. Save file as .mp4
    :param file_path: Path of the gif
    :param speed_mult: Speed multiplier
    :return:
    """

    clip = mp.VideoFileClip(file_path)
    sped_clip = clip.fx(mp.vfx.speedx, speed_mult)

    file, _ = os.path.splitext(file_path)

    sped_clip.write_videofile(f"{file}_{speed_mult}x.mp4")

    return
