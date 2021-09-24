import numpy as np
import tkinter as tk


class Drawing:
    """
    Draw image on board, return as vector [(x_1, y_1), ... , (x_n, y_n)]
    """
    def __init__(self, canvas_width: int = 500, canvas_height: int = 500):

        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

        self.master = tk.Tk()
        self.master.title("Draw an image")

        self.canvas = None
        self.prepare_canvas()

        self.points_m1 = (None, None)
        self.points_m2 = (None, None)

        self.finished = False

        self.xs = np.empty(0)
        self.ys = np.empty(0)

        tk.mainloop()

        return

    def prepare_canvas(self) -> None:
        """
        Create tk canvas, bind keys for drawing.
        """

        self.canvas = tk.Canvas(self.master,
                                width=self.canvas_width,
                                height=self.canvas_height)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)

        self.canvas.bind("<B1-Motion>", self.on_press)
        self.canvas.bind("<ButtonRelease>", self.on_release)

        return

    def on_press(self, event):
        """
        Event for clicking the canvas. Records position and draws line.

        :param event:
        :return:
        """
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

    def on_release(self, event) -> None:
        """
        Close window once drawing is finished

        :param event:
        :return:
        """
        self.canvas.unbind("<B1-Motion>")
        self.finished = True
        self.master.destroy()

        return
