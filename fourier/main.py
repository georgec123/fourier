import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.animation as animation
import imageio
import moviepy as mp


class Drawing:

    def __init__(self):

        self.master = tk.Tk()
        self.master.title("Draw an image")

        self.canvas = None

        self.points_m1 = (None, None)
        self.points_m2 = (None, None)

        self.finished = False

        self.xs = np.empty(0)
        self.ys = np.empty(0)

    def prepare_canvas(self, canvas_width=500, canvas_height=500):

        self.canvas = tk.Canvas(self.master,
                                width=canvas_width,
                                height=canvas_height)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)

        self.canvas.bind("<B1-Motion>", self.on_press)
        self.canvas.bind("<ButtonRelease>", self.on_release)

        return

    def on_press(self, event):

        if self.points_m1 == (None, None):
            self.points_m1 = (event.x, event.y)
            self.points_m2 = (event.x, event.y)

        x, y = event.x, event.y

        self.canvas.create_line(self.points_m2[0],
                                self.points_m2[1],
                                self.points_m1[0],
                                self.points_m1[1],
                                x,
                                y)

        self.points_m2 = self.points_m1
        self.points_m1 = (x, y)

        self.xs = np.append(self.xs, x)
        self.ys = np.append(self.ys, -1 * y)

        return

    def on_release(self, event):

        self.canvas.unbind("<B1-Motion>")
        self.finished = True
        self.master.destroy()

        return


class Fourier:

    def __init__(self, x: np.array, y: np.array, iters: int = 200):

        self.x = x
        self.y = y

        assert len(x) == len(y)

        self.compl = x + y * 1j

        self.line_len = len(x)
        self.dt = 1 / (1 + len(x))

        self.ts = np.linspace(0, 1, num=self.line_len)

        self.iter_list = np.linspace(-1 * iters, iters, 2 * iters + 1)

        self.coefs = self.get_fourier_coefs()

        self.gen_line = self.generate_new_line()

        return

    def get_fourier_coefs(self):

        fourier_coef_list = [self.get_fourier_coef(self.compl, self.ts, self.dt, n) for n in self.iter_list]

        return np.asarray(fourier_coef_list)

    @staticmethod
    def get_fourier_coef(f, ts, dt, n=0):

        coef = np.sum(f[1:] * dt * np.exp(-2 * np.pi * n * 1j * ts[1:]))

        return np.asarray(coef)

    def generate_new_line(self):
        line_list = [np.sum([coef[0] * np.exp(2 * np.pi * (coef[1]) * 1j * t)
                             for idx, coef in enumerate(zip(self.coefs, self.iter_list))])
                     for t in self.ts]
        line = np.asarray(line_list)
        shift_line = np.append(line[1:], line[0])[1:-1]
        # known bug

        return shift_line


def fourier_sum_term(coef, n, t):

    term = coef * np.exp(2 * np.pi * n * 1j * t)

    return term


def animate(info, line1, line2):
    idx = info[0]
    gen_line = info[2]
    if idx % 50 == 0:
        print(f"{idx}/{len(gen_line)}")

    gen_line = gen_line[:idx]

    gen_x = np.real(gen_line)
    gen_y = np.imag(gen_line)

    df_group = info[1]
    line1.set_data(df_group['x_cum'], df_group['y_cum'])
    line2.set_data(gen_x, gen_y)

    return line1, line2


def save_four_gif(fourier, save_path=r'C:\temp\myfirstAnimation.gif'):
    coef_df = pd.DataFrame({'coef': fourier.coefs, 'coef_index': fourier.iter_list})
    coef_df['size_coef'] = coef_df['coef'].apply(lambda x: np.abs(x))
    coef_df['key'] = 0
    time_df = pd.DataFrame({'t': fourier.ts})
    time_df['key'] = 0

    df = coef_df.merge(time_df, on='key', how='outer').drop(columns='key').sort_values(by=['t', 'size_coef'],
                                                                                       ascending=[True, False])
    df['f_term'] = fourier_sum_term(df['coef'], df['coef_index'], df['t'])

    df['x'] = df['f_term'].apply(lambda x: x.real)
    df['y'] = df['f_term'].apply(lambda x: x.imag)
    df[['x_cum', 'y_cum']] = df.groupby(by='t').cumsum()[['x', 'y']]

    min_max = df[['x_cum', 'y_cum']].agg(['min', 'max'])
    x_min, x_max = min_max.loc['min', 'x_cum'], min_max.loc['max', 'x_cum']
    y_min, y_max = min_max.loc['min', 'y_cum'], min_max.loc['max', 'y_cum']

    fig = plt.figure()
    ax = plt.axes(xlim=(x_min, x_max), ylim=(y_min, y_max))

    line1, = ax.plot([], [], lw=2)
    line2, = ax.plot([], [], lw=2)

    grouped = df.groupby(by='t')

    print('printing')
    ani = animation.FuncAnimation(fig, animate, frames=[(idx, d[1], fourier.gen_line) for idx, d in enumerate(grouped)],
                                  interval=1, blit=True, repeat=False, fargs=(line1, line2,))
    ani.save(save_path, fps=1e20)
    print('printed')


d = Drawing()
tk.mainloop()
pl = np.dstack((d.xs, d.ys))

four = Fourier(d.xs, d.ys, iters=200)

plt.plot(d.xs, d.ys)

plt.show()
plt.plot(four.gen_line.real, four.gen_line.imag)
plt.show()

save_four_gif(four)


gif_original = r'C:\temp\myfirstAnimation.gif'
gif_speed_up = r'C:\temp\myfirstAnimation2.gif'

gif = imageio.mimread(gif_original, memtest=False)

imageio.mimsave(gif_speed_up, gif, fps=100000)

clip = mp.VideoFileClip(r'C:\temp\myfirstAnimation.gif')

final = clip.fx(mp.vfx.speedx, 4)

final.write_videofile(r'C:\temp\myfirstAnimation.mp4')
