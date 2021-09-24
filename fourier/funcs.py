from xml.dom import minidom
from svg.path import parse_path
import numpy as np
import os
import pandas as pd
from fourier.Fourier import Fourier


def fourier_sum_term(coef: complex, n: int, t: float) -> complex:
    """
    Get the n'th sum member for a fourier transform

    :param coef: Series coefficient
    :param n: Coefficient index
    :param t: Parametric variable
    :return:
    """
    term = coef * np.exp(2 * np.pi * n * 1j * t)

    return term


def svg_to_vector(svg_path: str) -> np.array:
    """
    Process .svg file to numpy array of complex points

    :param svg_path: file path of svg file
    :return:
    """

    _, extension = os.path.splitext(svg_path)
    assert extension == 'svg', f"File type for svg_path should be 'svg', not {extension}"

    doc = minidom.parse(svg_path)

    path_strings = [path.getAttribute('d') for path
                    in doc.getElementsByTagName('path')]
    doc.unlink()

    points = []

    for path_string in path_strings:
        path = parse_path(path_string)
        for e in path:
            points.append(e.start)

    points = np.array(points)

    return points


def get_fourier_df_for_plotting(fourier: Fourier) -> pd.DataFrame:
    """
    Create long form dataframe for plotting
    :param fourier: Fourier object
    :return:
    """
    coef_df = pd.DataFrame({'coef': fourier.coefs, 'coef_index': fourier.iter_list})
    coef_df['size_coef'] = coef_df['coef'].apply(lambda x: np.abs(x))

    time_df = pd.DataFrame({'t': fourier.ts})

    coef_df['key'] = 0
    time_df['key'] = 0

    # create long df for group by
    df = coef_df.merge(time_df, on='key', how='outer').drop(columns='key').sort_values(by=['t', 'size_coef'],
                                                                                       ascending=[True, False])
    df['f_term'] = fourier_sum_term(df['coef'], df['coef_index'], df['t'])

    df['x'] = df['f_term'].apply(lambda x: x.real)
    df['y'] = df['f_term'].apply(lambda x: x.imag)
    df[['x_cum', 'y_cum']] = df.groupby(by='t').cumsum()[['x', 'y']]

    return df
