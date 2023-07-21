import io
import os
import random
import sys
import textwrap
import traceback
from copy import deepcopy

import matplotlib
import matplotlib.cm as cm
import matplotlib.font_manager
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as transforms
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator
from numpy.polynomial import Polynomial
from PIL import Image
from scipy.interpolate import BSpline, interp1d, make_interp_spline

matplotlib.use('Agg')
# matplotlib.font_manager._rebuild()


try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
except Exception as e:
    sys.path.append("/kaggle/input/gen-utils-easy")

try:
    from constants import RANDOM_LABELS, SYMBOLS, UNITS
    from generator_utils import (generate_series_name, get_random_equation,
                                 is_constant, is_numeric)
except ImportError:
    raise ImportError('Importing failed.')


def get_good_ticks(data, n_ticks):
    tick_min = np.amin(data)
    tick_max = np.amax(data)

    prune = None
    if random.random() >= 0.9:
        prune = random.choice(['lower', 'upper', 'both'])

    loc = MaxNLocator(
        nbins=n_ticks-1,
        prune=prune,
    )

    ticks = loc.tick_values(tick_min, tick_max)
    return ticks


def get_font_family():
    fonts = matplotlib.font_manager.findSystemFonts(
        fontpaths=None, fontext='ttf'
    )

    font_names = set()

    for font in fonts:
        name = matplotlib.font_manager.get_font(font).family_name
        font_names.add(name)
    font_names = list(font_names)
    return font_names


def get_formatter(input_list):
    input_list = deepcopy(input_list)
    input_range = max(input_list) - min(input_list)

    unit = random.choice(UNITS)
    currency_symbol = random.choice(SYMBOLS)

    formatter = ticker.ScalarFormatter()  # default formatter

    other_formatters = [
        ticker.FuncFormatter(lambda x, pos: f'{int(x)}%' if x.is_integer() else f'{x:.2f}%'),
        ticker.FuncFormatter(lambda x, pos: f'{int(x)}{unit}' if x.is_integer() else f'{x:.2f}{unit}'),
        ticker.FuncFormatter(lambda x, pos: f'{currency_symbol}{int(x)}' if x.is_integer() else f'{currency_symbol}{x:.2f}'),
    ]

    prob = random.random()
    if input_range >= 1e5:  # use integer formatting for large numbers
        formatter = ticker.FuncFormatter(lambda x, pos: f'{int(x)}')
    elif input_range <= 0.1:  # use float formatting for small numbers
        formatter = ticker.FuncFormatter(lambda x, pos: "{:.7f}".format(x).rstrip('0').rstrip('.'))
    elif (prob >= 0.975) & (input_range <= 1e5):  # use other formatters
        formatter = random.choice(other_formatters)
    return formatter


def add_legend_name(title_dict):
    p = random.random()

    if p <= 0.3:
        return title_dict['y_title']
    elif p <= 0.6:
        return random.choice(RANDOM_LABELS)
    else:
        return generate_series_name()


#########################################################################################
# Constants ---
#########################################################################################

FONT_FAMILY = [
    'DejaVu Sans',
    'Arial',
    'Times New Roman',
    'Courier New',
    'Helvetica',
    'Verdana',
    'Trebuchet MS',
    'Palatino Linotype',#Palatino
    'Georgia',
    'MesloLGS NF',
    'Lucida Grande',
    # 'Lucida Sans',
    # 'Source Code Pro',
]

# FONT_FAMILY = get_font_family()

MARKER_OPTIONS = [
    'o',
    's',
    'p',
    '*',
    '+',
    'x',
    'D',
    '.',
    '|',
    '_',
    '^',
    '<',
    '>',
]

COLOR_OPTIONS = [
    'b',
    'g',
    'r',
    'c',
    'm',
    'y',
    'k'
]

CMAP_OPTIONS = [
    'viridis',
    'inferno',
    'cool',
    'spring',
    'summer',
    'winter',
]

FONT_WEIGHTS = [
    'normal',
    'bold',
    'light',
    'ultralight',
    'heavy',
    'black',
    'semibold',
    'demibold',
]

SPINE_STYLES = [
    'center',
    'zero'
]

X_LIM_OPTIONS = [
    'tight',
    'semi-tight',
    'auto'
]

Y_LIM_OPTIONS = [
    'tight',
    'semi-tight',
    'zero',
    'sym',
    'unsym'
]

PLOT_OPTIONS = [
    'default',
    'new_y',
    'new_x',
    'new_xy',
    'interpolated'
]

SPECIAL_EFFECTS = [
    'shadow',
    'label_texts',
    'fill_vertical_region',
    'confidence_band',
    'annotate_max_min',
    'annotate_point',
    'random_equation',
    'random_text',
]

#########################################################################################
# PARAMS ---
#########################################################################################


def generate_random_params():
    # theme = 'ticks'
    theme = random.choices(
        ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks'],
        weights=[0.05, 0.05, 0.05, 0.05, 0.8],
        k=1
    )[0]

    # font weight ---
    if random.random() >= 0.75:
        font_weight = random.choice(FONT_WEIGHTS)
    else:
        font_weight = 'normal'

    # rotation ---
    rotation_x = random.choices(
        [0, 90, 270], weights=[0.90, 0.075, 0.025], k=1
    )[0]
    rotation_y = random.choices(
        [0, 45, 90],
        weights=[0.1, 0.1, 0.8], k=1
    )[0]

   # xlim and ylim approaches ---
    x_lim_approach = random.choices(
        X_LIM_OPTIONS, weights=[0.3, 0.3, 0.4], k=1
    )[0]

    y_lim_approach = random.choices(
        Y_LIM_OPTIONS, weights=[0.05, 0.10, 0.10, 0.50, 0.25], k=1
    )[0]

    plot_option = random.choices(
        PLOT_OPTIONS, weights=[0.25, 0.15, 0.25, 0.25, 0.10], k=1
    )[0]
    # ['default', 'new_y', 'new_x', 'new_xy', 'interpolated']

    params = {
        'sns_theme': theme,
        'plt_style': random.choice(plt.style.available),

        'font_family': random.choice(FONT_FAMILY),
        'font_size': random.randint(8, 12),
        'font_weight': font_weight,
        'font_color': random.choice(COLOR_OPTIONS),

        'span_width': random.randint(6, 24),

        'dpi': 100,
        'width': random.uniform(5.5, 8.0),
        'height': random.uniform(4, 6),

        'max_pixels_w': 1600,
        'max_pixels_h': 1200,

        'x_tick': random.random() >= 0.10,
        'y_tick': random.random() >= 0.10,

        'xtick_label_rotation': rotation_x,
        'ytick_label_rotation': rotation_y,
        'num_y_ticks': random.randint(4, 8),

        'aux_spine': random.random() >= 0.75,
        'left_spine': random.random() >= 0.05,

        'x_grid': random.random() >= 0.8,
        'y_grid': random.random() >= 0.8,
        'x_minor_grid': random.random() >= 0.9,
        'y_minor_grid': random.random() >= 0.9,

        'custom_spine': random.random() >= 0.95,
        'spine_style': random.choice(SPINE_STYLES),

        'x_lim_approach': x_lim_approach,
        'y_lim_approach': y_lim_approach,

        'plot_option': plot_option,
        'cmap_option': random.choice(CMAP_OPTIONS),
        'add_legend': random.random() >= 0.95,

        'add_special_effect': random.random() >= 0.8,
        'special_effect': random.choice(SPECIAL_EFFECTS)
    }
    return params

#########################################################################################
# Adjustments ---
#########################################################################################


class PixelAdjusterH:
    def __init__(self, params):
        params = deepcopy(params)
        self.params = params

        # main design variables ---
        self.font_size = params['font_size']
        self.max_y_tick_chars = params['max_y_tick_chars']
        self.rotation = params['ytick_label_rotation']
        self.n_ticks = params['num_y_ticks']
        self.max_pixels = params['max_pixels_h']
        self.soft_ub = params['height'] * params['dpi']
        self.height = params['height']

        if self.n_ticks >= 4:
            if random.random() >= 0.90:
                self.rotation = random.uniform(0, 45)
            else:
                self.rotation = 0
            self.max_y_tick_chars = 8

        if params['mean_y_chars'] > 4:
            if random.random() >= 0.90:
                self.rotation = random.uniform(0, 45)
            else:
                self.rotation = 0
            self.max_y_tick_chars = 8

        if self.n_ticks >= 8:
            self.font_size = random.randint(6, 10)
            self.height = max(6.0, self.height)

        # constants
        self.font_scale = 0.65
        self.offset_h = 2

        # flag to see if constraints are met
        self.stop = False

        # actions
        self.actions = ['decrease_n_ticks', 'decrease_font_size']

    def compute_pixels(self):
        pixels = (self.font_scale * self.font_size) * self.max_y_tick_chars * (self.n_ticks + self.offset_h)
        return pixels

    def decrease_n_ticks(self):
        while self.n_ticks > 3:  # requires minimum 3 ticks
            self.n_ticks -= 1
            pixels = self.compute_pixels()
            if pixels <= self.soft_ub:
                self.stop = True
                break

    def decrease_font_size(self):
        while self.font_size >= 8:  # requires minimum 8 font size
            self.font_size -= 1
            pixels = self.compute_pixels()
            if pixels <= self.soft_ub:
                self.stop = True
                break

    def get_adjusted_params(self):
        # shuffle actions ---
        random.shuffle(self.actions)

        # perform actions ---
        for action in self.actions:
            if action == 'decrease_n_ticks':
                self.decrease_n_ticks()
            if action == 'decrease_font_size':
                self.decrease_font_size()
            if self.stop:
                break

        if not self.stop:
            req_height = self.compute_pixels() / self.params['dpi']
            max_height = self.params['max_pixels_h'] / self.params['dpi']
            self.height = min(req_height, max_height)

        # return adjusted params ---
        self.params['height'] = self.height
        self.params['font_size'] = self.font_size
        self.params['num_y_ticks'] = self.n_ticks
        self.params['max_y_tick_chars'] = self.max_y_tick_chars
        self.params['ytick_label_rotation'] = self.rotation

        return self.params


class PixelAdjusterW:
    def __init__(self, params):
        params = deepcopy(params)
        self.params = params

        # main design variables ---
        self.font_size = params['font_size']
        self.max_x_tick_chars = params['max_x_tick_chars']
        self.rotation = params['xtick_label_rotation']

        self.n_points = params['n_points']
        self.max_pixels = params['max_pixels_w']
        self.soft_ub = params['width'] * params['dpi']
        self.width = params['width']

        # constants ---
        self.font_scale = 0.5
        self.offset_w = 2
        self.stop = False

        # update font size, width and rotation ---
        if (self.n_points >= 12) | (params['mean_x_chars'] >= 8) | (params['max_x_tick_chars'] >= 16):
            self.font_size = random.randint(6, 8)
            self.width = max(5.0, self.width)

            # handle rotation --
            if (random.random() >= 0.80) & (params['mean_x_chars'] <= 6):
                self.rotation = random.uniform(45, 90)
            else:
                self.rotation = 90

            self.max_x_tick_chars = 8

        if (params['max_x_tick_chars'] <= 4) & (self.n_points <= 8):
            self.rotation = 0
            self.params['font_size'] = max(8, self.params['font_size'])
            self.params['width'] = min(6, self.params['width'])

        # actions ---
        self.actions = ['decrease_font_size']

    def compute_pixels(self):
        pixels = (self.font_scale * self.font_size) * self.max_x_tick_chars * (self.n_points + self.offset_w)
        return pixels

    def decrease_font_size(self):
        while self.font_size >= 8:  # requires minimum
            self.font_size -= 1
            pixels = self.compute_pixels()
            if pixels <= self.soft_ub:
                self.stop = True
                break

    def get_adjusted_params(self):
        # shuffle actions ---
        random.shuffle(self.actions)

        # perform actions ---
        for action in self.actions:
            if action == 'decrease_font_size':
                self.decrease_font_size()
            if self.stop:
                break

        if not self.stop:
            req_width = max(7.0, self.compute_pixels() / self.params['dpi'])
            max_width = self.params['max_pixels_w'] / self.params['dpi']
            self.width = min(req_width, max_width)

        # return adjusted params ---
        self.params['width'] = self.width
        self.params['font_size'] = self.font_size
        self.params['max_x_tick_chars'] = self.max_x_tick_chars
        self.params['xtick_label_rotation'] = self.rotation

        return self.params


def adjust_params(params):
    params = deepcopy(params)
    if params['plt_style'] == 'dark_background':
        params['font_color'] = 'w'
        params['color'] = 'w'

    params = PixelAdjusterH(params).get_adjusted_params()
    params = PixelAdjusterW(params).get_adjusted_params()

    if params['font_size'] <= 8:
        params['span_width'] = 24

    # print(params['width'], params['height'])

    return params


#########################################################################################
# Data Processing ---
#########################################################################################


def get_smooth_data(x, y):
    x = deepcopy(x)
    y = deepcopy(y)

    n = len(x)
    x = np.arange(0, n)

    ns_min, ns_max = 5*n, 10*n

    if random.random() >= 0.25:
        x_smooth_start = - random.uniform(0.2, 0.8)
        x_smooth_end = n - 1 + random.uniform(0.2, 0.8)
        x_smooth = np.linspace(x_smooth_start, x_smooth_end, random.randint(ns_min, ns_max))
    else:
        x_smooth = np.linspace(0, n-1, random.randint(ns_min, ns_max))

    spl = make_interp_spline(x, y, k=3)  # type: BSpline
    y_smooth = spl(x_smooth)

    y_min = np.amin(y)
    y_max = np.amax(y)

    range_y = y_max - y_min
    sm_min, sm_max = y_min - range_y, y_max + range_y
    y_smooth = np.clip(y_smooth, sm_min, sm_max)

    return x_smooth, y_smooth


def get_noisy_data(x, y):
    x = deepcopy(x)
    y = deepcopy(y)

    clip_mag = (max(y) - min(y)) * 0.25

    n = len(x)
    x = np.arange(0, n)

    ns_min, ns_max = 8*n, 20*n

    if random.random() >= 0.25:
        x_new_start = - random.uniform(0.2, 0.8)
        x_new_end = n - 1 + random.uniform(0.2, 0.8)
        x_new = np.linspace(x_new_start, x_new_end, random.randint(ns_min, ns_max))

        # add points --
        y_min = np.amin(y)
        y_max = np.amax(y)
        pt1 = random.uniform(y_min, y_max)
        pt2 = random.uniform(y_min, y_max)
        x = [-1] + list(x) + [n]
        y = [pt1] + list(y) + [pt2]

    else:
        x_new = np.linspace(0, n-1, random.randint(ns_min, ns_max))

    f = interp1d(x, y)

    y_new = f(x_new)

    # Add noise to the data
    s = random.uniform(0.90, 0.99)
    e_scaling = np.random.uniform(1-s, 1 + s, len(x_new)) * y_new

    e = random.uniform(0.01, 0.10)
    noise = np.random.uniform(-e, e, len(x_new)) * e_scaling

    # clip the noise
    noise = np.clip(noise, a_min=-clip_mag, a_max=clip_mag)

    y_new = y_new + noise

    # Ensure y values at original x points coincide with original y values
    for i in range(len(x)):
        y_new[np.abs(x_new - x[i]) < 0.05] = y[i]

    # clip ---
    y_min = np.amin(y)
    y_max = np.amax(y)

    range_y = y_max - y_min
    sm_min, sm_max = y_min - range_y, y_max + range_y
    y_new = np.clip(y_new, sm_min, sm_max)

    return x_new, y_new


def get_interpolated_data(x, y):
    x = deepcopy(x)
    y = deepcopy(y)

    n = len(x)
    x = np.arange(0, n)

    x_new = []
    fractions = np.linspace(0, 1, random.randint(3, 5))

    for i in range(n-1):
        pts = [i + f for f in fractions]
        if i != 0:
            pts = pts[1:]
        x_new.extend(pts)

    f = interp1d(x, y)
    y_new = f(x_new)

    for i in range(len(x)):
        y_new[np.abs(x_new - x[i]) < 0.05] = y[i]

    return x_new, y_new


def get_data_with_new_categories(x, y, x_pre, x_post, max_post=4):
    x_copy = deepcopy(x)
    y_copy = deepcopy(y)

    x_post = x_post[:max_post]
    n_pre = len(x_pre)
    n_post = len(x_post)

    x_copy = x_pre + x_copy + x_post
    y_copy = [np.nan] * n_pre + y_copy + [np.nan] * n_post

    return x_copy, y_copy, n_pre, n_post


def parse_example(cfg, the_example):
    # get underlying data ---
    plot_title = the_example['plot_title']

    x_title = the_example['x_title']
    y_title = the_example['y_title']

    x_values = the_example['x_series']
    y_values = the_example['y_series']
    x_pre = []  # the_example['x_pre']
    x_post = []  # the_example['x_post']

    titles = {
        'x_title': x_title,
        'y_title': y_title,
        'plot_title': plot_title,
    }

    data_series = {
        'x_values': x_values,
        'y_values': y_values,
        'x_pre': x_pre,
        'x_post': x_post,
    }

    return titles, data_series

#########################################################################################
# Main ---
#########################################################################################


class BasicLinePlot:
    def __init__(self, cfg, the_example, texture_files=None, debug=False):
        self.cfg = cfg
        self.example = deepcopy(the_example)
        self.params = generate_random_params()
        self.debug = debug
        self.texture_files = texture_files

        # create data dependent params ---
        self.params['n_points'] = len(the_example['x_series'])

        self.params['max_x_tick_chars'] = max([len(str(x)) for x in the_example['x_series']])
        self.params['max_y_tick_chars'] = max([len(str(y)) for y in the_example['y_series']])

        # update max chars to reflect text wrapping ---
        self.params['max_x_tick_chars'] = min(self.params['max_x_tick_chars'], self.params['span_width'])

        self.params['mean_x_chars'] = np.mean([len(str(x)) for x in the_example['x_series']])
        self.params['mean_y_chars'] = np.mean([len(str(y)) for y in the_example['y_series']])

        # adjust params ---
        self.params = adjust_params(self.params)

        # stats ---
        self.num_x = len(the_example['x_series'])
        self.num_y = len(the_example['y_series'])
        self.y_min = min(the_example['y_series'])
        self.y_max = max(the_example['y_series'])
        self.x_tick_labels = deepcopy(the_example['x_series'])

        # configure style & create figure ---
        self.configure_style()
        self.fig, self.ax = self.get_figure_handles()

    def configure_style(self):
        plt.rcParams.update(plt.rcParamsDefault)  # reset to defaults

        sns.set_style(style=self.params['sns_theme'])
        plt.style.use(self.params['plt_style'])

        plt.rcParams['font.family'] = self.params['font_family']
        plt.rcParams['font.size'] = self.params['font_size']
        plt.rcParams['font.weight'] = self.params['font_weight']

        # tick parameters---
        tick_size = random.uniform(3.0, 6.0)
        tick_width = random.uniform(0.5, 1.5)

        tick_size_minor = random.uniform(2.0, tick_size)
        tick_width_minor = random.uniform(0.5, tick_width)

        plt.rcParams['xtick.major.size'] = tick_size
        plt.rcParams['ytick.major.size'] = tick_size

        plt.rcParams['xtick.major.width'] = tick_width
        plt.rcParams['ytick.major.width'] = tick_width

        plt.rcParams['xtick.minor.size'] = tick_size_minor
        plt.rcParams['ytick.minor.size'] = tick_size_minor

        plt.rcParams['xtick.minor.width'] = tick_width_minor
        plt.rcParams['ytick.minor.width'] = tick_width_minor

    def get_figure_handles(self):
        width = self.params['width']
        height = self.params['height']
        dpi = self.params['dpi']

        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        return fig, ax

    # -----------------------------------------------------------------------------------#
    # error bars
    # -----------------------------------------------------------------------------------#
    def plot_error_bars(self):
        if (random.random() <= 0.2) & (self.params['plot_option'] == 'default'):
            e_scaling = np.random.uniform(0.9, 1.1, len(self.x_values)) * np.absolute(self.y_values)
            y_errors = np.random.uniform(0.05, 0.1, len(self.x_values)) * e_scaling

            # clipping of error bar
            clip_val = (max(self.y_values) - min(self.y_values))*0.1
            y_errors = np.clip(y_errors, 0, clip_val)

            self.ax.errorbar(
                np.arange(0, len(self.x_values)),
                self.y_values,
                yerr=y_errors,
                fmt='none',  # 'none',
                ecolor=random.choice(COLOR_OPTIONS),
                elinewidth=random.uniform(0.8, 1.5),
                capsize=random.uniform(3, 5),
                barsabove=random.choice([True, False]),
                errorevery=random.randint(1, 2),
                alpha=random.uniform(0.25, 1.0),
            )

    # -----------------------------------------------------------------------------------#
    # axis limits
    # -----------------------------------------------------------------------------------#
    def set_axis_limits(self):
        # data ranges - x axis
        x_min, x_max = 0, self.num_x - 1
        y_min, y_max = self.y_min, self.y_max

        y_range = y_max - y_min
        if y_range < 1e-6:  # constant line
            y_range = random.uniform(0.1, 1.0)

        # --- set x axis limits ---------------------------------------------------------#
        approach = self.params['x_lim_approach']
        buffer = 0.02

        if approach == 'tight':
            x_lb = x_min - buffer
            x_ub = x_max + buffer

        if approach == 'semi-tight':
            if random.random() >= 0.5:
                x_lb = x_min - buffer
                x_ub = x_max + random.uniform(0.1, 1)
            else:
                x_lb = x_min - random.uniform(0.1, 1)
                x_ub = x_max + buffer

        if approach == 'auto':
            x_lb = x_min - random.uniform(0.05, 0.2)
            x_ub = x_max + random.uniform(0.05, 0.2)
        self.ax.set_xlim(x_lb, x_ub)

        # --- set y axis limits ---------------------------------------------------------#
        approach = self.params['y_lim_approach']
        rs_low, rs_high = 0.05, 0.10

        if approach == 'tight':
            y_lb = y_min - random.uniform(0.01, 0.02) * y_range
            y_ub = y_max + random.uniform(0.01, 0.02) * y_range

        if approach == 'semi-tight':
            if random.random() >= 0.5:
                y_lb = y_min - random.uniform(0.01, 0.02) * y_range
                y_ub = y_max + random.uniform(rs_low, rs_high) * y_range
            else:
                y_lb = y_min - random.uniform(rs_low, rs_high) * y_range
                y_ub = y_max + random.uniform(0.01, 0.02) * y_range

        if approach == 'zero':
            y_lb = y_min - random.uniform(0.1, 0.2) * y_range
            y_ub = y_max + random.uniform(0.1, 0.2) * y_range

            if (y_lb >= 0) & ((y_max - 0.5 * y_range) <= 0):
                y_lb = 0

            if (y_ub <= 0) & ((y_min + 0.5 * y_range) >= 0):
                y_ub = 0

        if approach == 'sym':
            fraction = random.uniform(rs_low, rs_high)
            y_lb = y_min - fraction * y_range
            y_ub = y_max + fraction * y_range

        if approach == 'unsym':
            y_lb = y_min - random.uniform(rs_low, rs_high) * y_range
            y_ub = y_max + random.uniform(rs_low, rs_high) * y_range
        self.ax.set_ylim(y_lb, y_ub)

    # -----------------------------------------------------------------------------------#
    # ticks
    # -----------------------------------------------------------------------------------#
    def configure_ticks(self):
        # tick parameters --
        direction = random.choice(['in', 'out', 'inout'])
        x_top = (random.random() >= 0.6) & (self.params['aux_spine'])
        x_labeltop = (random.random() >= 0.8) & (x_top) & (self.params['max_x_tick_chars'] <= 5)
        self.x_labeltop = x_labeltop

        y_left = (self.params['left_spine']) | (random.random() >= 0.98)
        y_labelleft = y_left

        y_right = (random.random() >= 0.6) & (self.params['aux_spine'])
        y_labelright = (random.random() >= 0.8) & (y_right)

        if not y_left:
            y_right = True
            y_labelright = True

        self.x_pad = random.uniform(4, 7)
        self.y_pad = random.uniform(4, 7)

        self.ax.minorticks_on()  # turn on minor ticks

        # set x ticks --------
        x_tick_positions = np.arange(self.num_x)  # uniform x ticks
        self.ax.set_xticks(x_tick_positions)
        x_tick_labels = [
            "\n".join(textwrap.wrap(cat, width=self.params['span_width'])) for cat in self.x_tick_labels
        ]
        self.ax.set_xticklabels(x_tick_labels, minor=False)  # set x tick labels

        if (self.params['max_x_tick_chars'] <= 6) & (random.random() >= 0.25):
            self.custom_align_xtick_labels()

        self.ax.tick_params(
            axis='x',
            which='both',
            rotation=self.params['xtick_label_rotation'],
            direction=direction,
            top=x_top,  # ---
            labeltop=x_labeltop,
            zorder=5,
            pad=self.x_pad,

        )

        if not is_numeric(self.x_values):
            self.ax.xaxis.set_tick_params(which='minor', bottom=False, top=False)
        elif random.random() >= 0.25:
            self.ax.xaxis.set_tick_params(which='minor', bottom=False, top=False)

        # set y ticks --------
        y_data = deepcopy(self.y_values)
        y_data.append(self.ax.get_ylim()[0])
        y_data.append(self.ax.get_ylim()[1])

        try:
            y_tick_positions = get_good_ticks(y_data, n_ticks=self.params['num_y_ticks'])
            self.ax.set_yticks(y_tick_positions)
        except ZeroDivisionError:
            print('ZeroDivisionError')

        y_formatter = get_formatter(self.y_values)
        self.ax.yaxis.set_major_formatter(y_formatter)

        # tick labels alignment --
        if (self.params['num_y_ticks'] <= 5) & (random.random() >= 0.9):
            self.custom_align_ytick_labels()

        self.ax.tick_params(
            axis='y',
            which='both',
            rotation=self.params['ytick_label_rotation'],
            direction=direction,
            left=y_left,  # ---
            right=y_right,  # ---
            labelleft=y_labelleft,
            labelright=y_labelright,
            zorder=5,
            pad=self.y_pad,
        )
        if random.random() >= 0.25:
            self.ax.yaxis.set_tick_params(which='minor', left=False, right=False)

    def custom_align_xtick_labels(self):
        halign = random.choice(['right', 'left', 'center'])
        valign = random.choice(['top', 'bottom', 'center'])

        if valign == 'bottom':  # inward
            self.x_pad = - (10 + self.params['height'])
            if self.params['x_grid']:
                halign = 'right'

        elif valign == 'top':
            self.x_pad = (5 + self.params['height'])
        else:  # center
            self.x_pad = (10 + self.params['height'])

        for tick in self.ax.xaxis.get_major_ticks():
            tick.label1.set_horizontalalignment(halign)
            tick.label1.set_verticalalignment(valign)

    def custom_align_ytick_labels(self):
        halign = random.choice(['right', 'left', 'center'])
        valign = random.choice(['top', 'bottom', 'center'])

        if halign == 'left':  # inward
            self.y_pad = - (10 + self.params['width'])
            if self.params['y_grid']:
                valign = 'top'

        elif halign == 'right':
            self.y_pad = (5 + self.params['width'])
        else:  # center
            self.y_pad = (10 + self.params['width'])

        for tick in self.ax.yaxis.get_major_ticks():
            tick.label1.set_horizontalalignment(halign)
            tick.label1.set_verticalalignment(valign)

    # -----------------------------------------------------------------------------------#
    # gridlines
    # -----------------------------------------------------------------------------------#

    def configure_gridlines(self):
        # ----
        major_linestyle = random.choice(['-', '--', ':', '-.'])
        minor_linestyle = random.choice(['-', '--', ':', '-.'])

        major_linewidth = random.uniform(0.5, 1.0)
        minor_linewidth = random.uniform(0.1, major_linewidth)

        major_color = random.choice(COLOR_OPTIONS)
        minor_color = random.choice(COLOR_OPTIONS)

        major_alpha = random.uniform(0.4, 1.0)
        minor_alpha = random.uniform(0.1, major_alpha)

        # set x axis gridlines ---
        if self.params['x_grid']:

            self.ax.xaxis.grid(
                True,
                linestyle=major_linestyle,
                linewidth=major_linewidth,
                color=major_color,
                alpha=major_alpha,
                which='major',
            )
            if self.params['x_minor_grid']:
                self.ax.xaxis.grid(
                    True,
                    linestyle=minor_linestyle,
                    linewidth=minor_linewidth,
                    color=minor_color,
                    alpha=minor_alpha,
                    which='minor',
                )
        else:
            self.ax.xaxis.grid(visible=False, which='both')

        # set y axis gridlines ---
        if self.params['y_grid']:
            self.ax.yaxis.grid(
                True,
                linestyle=major_linestyle,
                linewidth=major_linewidth,
                color=major_color,
                alpha=major_alpha,
                which='major',
            )
            if self.params['y_minor_grid']:
                self.ax.yaxis.grid(
                    True,
                    linestyle=minor_linestyle,
                    linewidth=minor_linewidth,
                    color=minor_color,
                    alpha=minor_alpha,
                    which='minor',
                )
            # remove gridlines too close to spines
            self.remove_closest_gridline()
        else:
            self.ax.yaxis.grid(visible=False, which='both')

    def remove_closest_gridline(self, threshold=0.05):
        # Get limits
        ylim = self.ax.get_ylim()
        yrange = ylim[1] - ylim[0]

        # Get gridlines and ticks
        ygridlines = self.ax.yaxis.get_gridlines()
        yticks = self.ax.get_yticks()

        for i, tick in enumerate(yticks):
            if (abs(tick - ylim[0]) / yrange < threshold) & (abs(tick - ylim[0]) > 1e-3):
                ygridlines[i].set_visible(False)

            if (abs(tick - ylim[1]) / yrange < threshold) & (abs(tick - ylim[1]) > 1e-3):
                ygridlines[i].set_visible(False)

    # ------------------------------------------------------------------------------------#
    # titles
    # ------------------------------------------------------------------------------------#

    def configure_titles(self):
        # bounding box ---
        box_color = random.choice(['c', 'k', 'gray', 'r', 'b', 'g', 'm', 'y'])
        padding = random.uniform(0, 1)
        boxstyle = random.choice(['round', 'square'])

        if random.random() >= 0.90:
            bbox_props = {
                'facecolor': box_color,
                'alpha': random.uniform(0.1, 0.2),
                'pad': padding,
                'boxstyle': boxstyle
            }
        else:
            bbox_props = None

        # font dict --
        if self.params['font_color'] == 'y':  # no yellow font color
            self.params['font_color'] = 'k'

        font_dict = {
            'family': self.params['font_family'],
            'color': self.params['font_color'],
            'weight': self.params['font_weight'],
            'size': self.params['font_size'],
        }

        axes_font_dict = deepcopy(font_dict)
        axes_font_dict['size'] = self.params['font_size'] + random.randint(0, 4)

        x_location = 'center'
        y_location = 'center'

        if random.random() >= 0.85:
            x_location = random.choice(['left', 'right', 'center'])
            y_location = random.choice(['top', 'bottom', 'center'])

        if (self.params['custom_spine']) & (self.params['spine_style'] == 'center'):
            x_location = 'right'
            y_location = 'top'

        if random.random() >= 0.05:
            self.ax.set_xlabel(
                self.title_dict['x_title'],
                fontdict=axes_font_dict,
                loc=x_location,
                labelpad=random.uniform(4, 10),
                bbox=bbox_props,
            )

            self.ax.set_ylabel(
                self.title_dict['y_title'],
                fontdict=axes_font_dict,
                loc=y_location,
                labelpad=random.uniform(4, 10),
                bbox=bbox_props,
            )

        # figure title ---
        fig_width_inch = self.fig.get_figwidth()
        fig_dpi = self.fig.dpi
        fig_width_pixel = fig_width_inch * fig_dpi
        char_width_pixel = 12
        num_chars = int(fig_width_pixel / char_width_pixel)

        # Set a long title and use textwrap to automatically wrap the text
        title = '\n'.join(textwrap.wrap(self.title_dict['plot_title'], num_chars))
        title_font_dict = deepcopy(font_dict)
        title_font_dict['size'] = self.params['font_size'] + random.randint(2, 4)
        title_loc = random.choice(['left', 'right', 'center'])

        if (self.params['custom_spine']) & (self.params['spine_style'] == 'center'):
            title_loc = random.choice(['left', 'right'])

        if self.x_labeltop:
            title_pad = - 32
        elif self.params['aux_spine']:
            title_pad = 24
        else:
            title_pad = random.randint(6, 24)
            if random.random() >= 0.8:
                title_pad = - title_pad

        if random.random() >= 0.05:
            self.ax.set_title(
                title,
                fontdict=title_font_dict,
                loc=title_loc,
                pad=title_pad,
                bbox=bbox_props
            )

    # -----------------------------------------------------------------------------------#
    # legend
    # -----------------------------------------------------------------------------------#

    def configure_legend(self):
        # --------------------------------
        font_dict = {
            'family': self.params['font_family'],
            'style': random.choice(['normal', 'italic']),
            'variant': random.choice(['normal', 'small-caps']),
            'weight': self.params['font_weight'],
            'stretch': random.choice(['normal', 'condensed', 'expanded']),
            'size': self.params['font_size'] + random.randint(1, 2),
        }

        if self.params['add_legend']:
            if random.random() > 0.1:
                legend = self.ax.legend(prop=font_dict, loc='best', frameon=True)
                frame = legend.get_frame()         # frame.set_facecolor('red')
                frame.set_edgecolor(random.choice(COLOR_OPTIONS))
            else:
                self.ax.legend(prop=font_dict, loc='center left', bbox_to_anchor=(1, 0.5))

    # -----------------------------------------------------------------------------------#
    # spine
    # -----------------------------------------------------------------------------------#
    def configure_spine(self):
        # spine visibility ---
        active_spines = ['left', 'bottom', 'right', 'top']

        self.ax.spines['left'].set_visible(True)
        self.ax.spines['bottom'].set_visible(True)

        if not self.params['aux_spine']:
            self.ax.spines['right'].set_visible(False)
            self.ax.spines['top'].set_visible(False)
            active_spines.remove('right')
            active_spines.remove('top')

        if not self.params['left_spine']:
            self.ax.spines['left'].set_visible(False)
            self.ax.spines['right'].set_visible(True)

            active_spines.remove('left')
            active_spines.append('right')

        # set linewidth ---
        linewidth = random.uniform(1.0, 3.0)
        for spine in active_spines:
            self.ax.spines[spine].set_linewidth(linewidth)

        # set color ---
        color = random.choice(['r', 'g', 'b', 'k', 'gray'])
        if random.random() >= 0.75:
            color = 'k'
        for spine in active_spines:
            self.ax.spines[spine].set_color(color)

        # set spine positions ---
        picked_style = self.params['spine_style']

        if self.params['custom_spine'] & (not self.params['aux_spine']):  # apply styles
            self.params['x_grid'] = False
            self.params['y_grid'] = False
            self.params['y_minor_grid'] = False
            self.params['num_y_ticks'] = min(4, self.params['num_y_ticks'])

            if picked_style == 'zero':
                if (np.amin(self.y_values) < 0) & (np.amax(self.y_values) > 0):
                    self.ax.spines[active_spines].set_position('zero')

            if picked_style == 'center':
                self.ax.spines[active_spines].set_position('center')

    # -----------------------------------------------------------------------------------#
    # Plot
    # -----------------------------------------------------------------------------------#

    def make_basic_plot(self, x, y, force_no_marker=False, force_marker=False):
        x = deepcopy(x)
        y = deepcopy(y)

        # marker -----
        marker = random.choices(
            [random.choice(MARKER_OPTIONS), None], weights=[0.75, 0.25], k=1
        )[0]

        if force_no_marker:
            marker = None

        if force_marker:
            marker = random.choice(['*', '+', 'x', '^'])

        # color ----
        color = 'k'
        if random.random() >= 0.5:
            color = random.choice(COLOR_OPTIONS)

        if self.params['plt_style'] == 'dark_background':
            color = 'w'

        # linestyle ---
        linestyle = random.choices(
            ['-', '--', '-.'], weights=[0.90, 0.05, 0.05], k=1
        )[0]

        if linestyle == '-':
            linewidth = random.uniform(0.7, 2.0)
        else:
            linewidth = random.uniform(2.0, 3.0)

        # numeric x handling
        x_ = np.arange(len(x))

        line, = self.ax.plot(
            x_,
            y,
            color=color,
            linestyle=linestyle,
            marker=marker,
            linewidth=linewidth,
            markersize=random.uniform(2, 5) if marker is not None else None,
            solid_capstyle=random.choice(['butt', 'round', 'projecting']),
            solid_joinstyle=random.choice(['miter', 'round', 'bevel']),
            alpha=random.uniform(0.8, 1),
            label=add_legend_name(self.title_dict),
        )

    def fancy_plot_v1(self, x, y, **kwargs):
        x = deepcopy(x)
        x = np.arange(len(x))
        y = deepcopy(y)

        # Change the color and linewidth of line segments based on the rate of change in y values
        for i in range(1, len(y)):
            dy = y[i] - y[i - 1]
            color = 'red' if dy < 0 else 'green'
            linewidth = min(max(1, abs(dy) / max(y) * 50), 5)
            self.ax.plot(x[i-1:i+1], y[i-1:i+1], color=color, linewidth=linewidth, zorder=3)

    def fancy_plot_v2(self, x, y, **kwargs):
        x = deepcopy(x)
        x = np.arange(len(x))
        y = deepcopy(y)

        # Layered line plot (glow effect)
        self.ax.plot(x, y, linewidth=15, color='orange', alpha=0.1, zorder=1)
        self.ax.plot(x, y, linewidth=10, color='orange', alpha=0.2, zorder=2)
        self.ax.plot(x, y, linewidth=5, color='orange', alpha=0.3, zorder=3)

        line, = self.ax.plot(x, y, linewidth=2, color='black', zorder=4, label=add_legend_name(self.title_dict))
        line.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])

        self.ax.fill_between(x, y, color='orange', alpha=0.1, zorder=2)
        self.ax.fill_between(x, y, color='yellow', alpha=0.05, zorder=3)

    def fancy_plot_v3(self, x, y, **kwargs):
        x = deepcopy(x)
        x = np.arange(len(x))
        y = deepcopy(y)

        p = Polynomial.fit(range(len(x)), y, deg=3)
        dp = p.deriv()

        self.ax.plot(x, y, color='black', zorder=2, label=add_legend_name(self.title_dict))
        # Annotate the line with arrows indicating the trend direction
        for i in range(1, len(x) - 1):
            trend = dp(i)
            color = 'red' if trend < 0 else 'green'
            self.ax.annotate(
                '',
                xy=(x[i], y[i]),
                xytext=(x[i-1], y[i-1]),
                arrowprops=dict(
                    facecolor=color,
                    edgecolor=color,
                    shrink=0.05,
                    width=1,
                    headwidth=5),
                zorder=3)

    def fancy_plot_v4(self, x, y, **kwargs):
        x = deepcopy(x)
        x = np.arange(len(x))
        y = deepcopy(y)

        lc = LinearSegmentedColormap.from_list('gradient', ['cyan', 'purple'], N=len(y))
        for i in range(1, len(y)):
            self.ax.plot(x[i-1:i+1], y[i-1:i+1], marker='o', linestyle='-', color=lc(i/len(y)))

    # -----------------------------------------------------------------------------------#
    # invert y axis
    # -----------------------------------------------------------------------------------#
    def invert_y_axis(self):
        if (random.random() >= 0.0) & (np.amin(self.y_values) * np.amax(self.y_values) > 0):
            self.ax.invert_yaxis()
            print('inverted y axis')

    # -----------------------------------------------------------------------------------#
    # drop lines
    # -----------------------------------------------------------------------------------#

    def add_drop_line(self):
        if random.random() >= 0.95:
            # Draw horizontal lines from x axis to each point
            x = np.arange(len(self.y_values))
            y = deepcopy(self.y_values)
            y_min = np.min(y)

            for i, j in zip(x, y):
                self.ax.plot([i, i], [y_min, j], color='red', linestyle='-', linewidth=random.uniform(0.15, 0.3))

    # -----------------------------------------------------------------------------------#
    # annotate each point
    # -----------------------------------------------------------------------------------#
    def annotate_each_point(self):
        if random.random() >= 0.95:
            # Draw horizontal lines from x axis to each point
            x = np.arange(len(self.y_values))
            y = deepcopy(self.y_values)

            p = random.random()

            def format_text(i, j):
                if p <= 0.1:
                    return f'({i})'
                elif p <= 0.2:
                    return f'{round(j, 2)}'
                elif p <= 0.3:
                    return random.choice(['+', '-', '*', '/'])
                elif p <= 0.4:
                    return random.choice(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
                elif p <= 0.5:
                    return f'{i}'
                elif p <= 0.6:
                    return f'[{i}]'
                else:
                    return str(i)

            # Highlight each point on the line
            for i, j in zip(x, y):
                self.ax.annotate(
                    format_text(i, j),
                    (i, j),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=8,
                    color=random.choice(['black', 'red']),
                )

    # -----------------------------------------------------------------------------------#
    # stats
    # -----------------------------------------------------------------------------------#

    def add_stat_info(self):
        if random.random() >= 0.95:
            # Add statistical information
            avg = np.mean(self.y_values)
            min_val = round(np.min(self.y_values), 2)
            max_val = round(np.max(self.y_values), 2)

            stats_text = f'Average: {avg:.2f}\nMin: {min_val}\nMax: {max_val}'

            self.ax.text(
                random.uniform(0.02, 0.2),
                random.uniform(0.8, 0.95),
                stats_text,
                transform=self.ax.transAxes,
                verticalalignment='top',
                bbox=dict(
                    boxstyle='round',
                    facecolor='white',
                    alpha=random.uniform(0.2, 0.5),
                )
            )
    # -----------------------------------------------------------------------------------#
    # relief
    # -----------------------------------------------------------------------------------#

    def add_shaded_relief(self):
        if random.random() >= 0.95:
            grid_x, grid_y = np.meshgrid(
                np.linspace(0, 1, 1000), np.linspace(0, 1, 1000)
            )
            z = grid_y + np.interp(grid_x, np.linspace(0, 1, len(self.y_values)), self.y_values) / max(self.y_values)
            self.ax.imshow(z, extent=(0, 1, 0, 1), origin='lower', cmap='terrain',
                           alpha=0.5, aspect='auto', transform=self.ax.transAxes)

    # -----------------------------------------------------------------------------------#
    # inset
    # -----------------------------------------------------------------------------------#

    def add_inset(self):
        if random.random() >= 0.97:
            # create an inset axes for a zoomed in plot
            w = random.uniform(0.1, 0.4)
            axins = self.ax.inset_axes([0.5, 0.5, w, w])
            x_ = np.arange(len(self.x_values))
            axins.plot(x_, self.y_values, color='blue', linewidth=1)
            axins.set_xticklabels('')
            axins.set_yticklabels('')

    def add_inset_pie(self):
        if random.random() >= 0.99:
            # Create an inset axes for a pie chart of positive and negative y-values
            w = random.uniform(0.1, 0.4)
            axins = self.ax.inset_axes([0.1, 0.6, w, w])
            pos_values = sum(1 for _ in filter(lambda x: x >= 0, self.y_values))
            neg_values = len(self.y_values) - pos_values
            axins.pie([pos_values, neg_values], labels=['', ''], autopct='%1.1f%%')

    def add_inset_zoom(self):
        if random.random() >= 0.99:
            # Create an inset axes for a zoomed in part of the line
            w = random.uniform(0.1, 0.2)
            axins = self.ax.inset_axes([random.uniform(0.2, 0.8), random.uniform(0.2, 0.8), w, w])
            start = random.randint(0, len(self.x_values) // 2)
            end = random.randint(start, len(self.x_values))
            axins.plot(self.x_values[start:end], self.y_values[start:end], color='blue', linewidth=1)
            axins.set_xticklabels([])
            axins.set_yticklabels([])

    # -----------------------------------------------------------------------------------#
    # slope
    # -----------------------------------------------------------------------------------#

    def add_slope_indicator(self):
        if random.random() >= 0.98:
            # Choose a random point to calculate the slope
            index = random.randint(1, len(self.x_values) - 2)
            dx = 1
            dy = self.y_values[index + 1] - self.y_values[index - 1]
            slope = dy / dx

            # Create the arrow representing the slope
            arrowprops = dict(facecolor='red', edgecolor='red', arrowstyle='-|>', linewidth=1)
            self.ax.annotate(
                '',
                xy=(index, self.y_values[index]),
                xytext=(index - 0.5 * dx, self.y_values[index] - 0.5 * dy),
                arrowprops=arrowprops
            )

            # Create the text indicating the slope value
            self.ax.text(
                index,
                self.y_values[index],
                f'Slope: {slope:.2f}',
                fontsize=random.randint(6, 8),
                ha='right',
                color='red'
            )

    # -----------------------------------------------------------------------------------#
    # horizontal region
    # -----------------------------------------------------------------------------------#

    def highlight_horizontal_region(self):
        # Highlight a random range on x-axis
        if random.random() >= 0.98:
            color = random.choice(COLOR_OPTIONS)
            start = random.randint(0, len(self.x_values) // 2)
            end = random.randint(start, len(self.x_values))

            self.ax.axvspan(
                start, end,
                facecolor=color,
                alpha=random.uniform(0.1, 0.3)
            )

    # -----------------------------------------------------------------------------------#
    # add a shape
    # -----------------------------------------------------------------------------------#
    def add_shape(self):
        # Draw a circle around the maximum y value
        if random.random() >= 0.95:
            max_y = max(self.y_values)
            max_x = self.y_values.index(max_y)
            circle = Circle((max_x, max_y), 0.5, fill=False, edgecolor='red', linewidth=2)
            self.ax.add_patch(circle)

    # -----------------------------------------------------------------------------------#
    # fill vertical region
    # -----------------------------------------------------------------------------------#
    def fill_vertical_region(self):
        # Define the threshold value
        # print('filling vertical region')
        if random.random() >= 0.98:
            color = random.choice(COLOR_OPTIONS)
            threshold = np.percentile(self.y_values, random.randint(20, 80))

            if random.random() >= 0.9:
                self.ax.axhline(
                    threshold,
                    color=color,
                    lw=random.uniform(0.5, 2.5),
                    alpha=random.uniform(0.5, 0.75)
                )

            x_ = np.arange(len(self.y_values))
            self.ax.fill_between(
                x_, 0, 1,
                where=self.y_values > threshold,
                color=color,
                alpha=random.uniform(0.25, 0.5),
                transform=self.ax.get_xaxis_transform(),
            )

    # -----------------------------------------------------------------------------------#
    # add confidence band
    # -----------------------------------------------------------------------------------#
    def add_confidence_band(self):
        if random.random() >= 0.95:
            y_range = np.amax(self.y_values) - np.amin(self.y_values)
            band_width = random.uniform(0.02, 0.10) * y_range
            y_upper = self.y_values + band_width
            y_lower = self.y_values - band_width

            x_ = np.arange(len(self.y_values))

            self.ax.fill_between(
                x_, y_lower, y_upper,
                alpha=random.uniform(0.05, 0.2),
                color=random.choice(COLOR_OPTIONS)
            )

    # -----------------------------------------------------------------------------------#
    # annotate min/max
    # -----------------------------------------------------------------------------------#
    def annotate_max_min(self):
        if random.random() >= 0.95:
            max_idx = np.argmax(self.y_values)
            min_idx = np.argmin(self.y_values)
            n = len(self.y_values)

            x_ = np.arange(len(self.y_values))

            max_point = (x_[max_idx], self.y_values[max_idx])
            min_point = (x_[min_idx], self.y_values[min_idx])

            color = random.choice(COLOR_OPTIONS)

            max_text = random.choice(['maximum', 'max', 'peak', 'maximum value'])
            if random.random() >= 0.5:
                max_text = generate_series_name()

            min_text = random.choice(['minimum', 'min', 'trough', 'minimum value'])
            if random.random() >= 0.5:
                max_text = generate_series_name()

            self.ax.annotate(
                max_text,
                xy=max_point,
                xytext=(30, 30),
                arrowprops=dict(facecolor=color, shrink=0.1, width=3, headwidth=8),
                textcoords='offset points',
                horizontalalignment='right' if max_idx > n/2 else 'left',
                verticalalignment='top',
            )

            self.ax.annotate(
                min_text,
                xy=min_point,
                xytext=(30, -30),
                arrowprops=dict(facecolor=color, shrink=0.1, width=3, headwidth=8),
                textcoords='offset points',
                horizontalalignment='right' if min_idx > n/2 else 'left',
                verticalalignment='bottom',
            )

    # -----------------------------------------------------------------------------------#
    # random equation
    # -----------------------------------------------------------------------------------#
    def add_random_equation(self):
        if random.random() >= 0.95:
            equation = get_random_equation()

            self.ax.text(
                random.uniform(0.1, 0.9), random.uniform(0.1, 0.9), equation,
                fontsize=random.randint(10, 16),
                color=random.choice(COLOR_OPTIONS),
                alpha=random.uniform(0.5, 0.95),
                transform=self.ax.transAxes,
            )

    # -----------------------------------------------------------------------------------#
    # random text
    # -----------------------------------------------------------------------------------#
    def add_random_text(self):
        if random.random() >= 0.95:
            t1 = generate_series_name()
            t2 = random.choice(RANDOM_LABELS)
            t3 = random.choice(RANDOM_LABELS)

            text = f'{t1}: {t2} {t3}'
            text = "\n".join(textwrap.wrap(text, width=random.randint(16, 32)))

            if random.random() >= 0.5:
                self.ax.text(
                    random.uniform(0.7, 0.9), random.uniform(0.1, 0.4), text,
                    fontsize=random.randint(8, 10),
                    color=random.choice(COLOR_OPTIONS),
                    alpha=random.uniform(0.2, 0.9),
                    transform=self.ax.transAxes,
                    # rotation=random.choice([0, 90, 270]),
                )
            else:
                self.ax.text(
                    random.uniform(0.5, 0.7), random.uniform(0.1, 0.4), text,
                    fontsize=random.randint(8, 10),
                    color=random.choice(COLOR_OPTIONS),
                    alpha=random.uniform(0.2, 0.9),
                    transform=self.ax.transAxes,
                    # rotation=random.choice([0, 90, 270]),
                    bbox=dict(
                        boxstyle='round,pad=0.5',
                        fc=f'{random.choice(COLOR_OPTIONS)}',
                        ec='k',
                        lw=random.uniform(0.9, 1.9),
                        alpha=random.uniform(0.2, 0.5),
                    )
                )

    # -----------------------------------------------------------------------------------#
    # plot creation
    # -----------------------------------------------------------------------------------#
    def create_plot(self):
        p = random.random()
        if p >= 0.75:
            self.make_basic_plot(self.x_values, self.y_values)
        else:
            fn = random.choice(
                [self.fancy_plot_v1, self.fancy_plot_v2, self.fancy_plot_v3, self.fancy_plot_v4]
            )
            fn(self.x_values, self.y_values)

    # ------------------------------------------------------------------------------------#
    # main api
    # ------------------------------------------------------------------------------------#

    def make_line_plot(self, graph_id):
        """main function to make a line plot

        :param the_example: underlying data
        :type the_example: dict
        """
        try:
            the_example = deepcopy(self.example)
            title_dict, data_series = parse_example(self.cfg, the_example)
            self.title_dict = title_dict
            self.x_values = data_series['x_values']
            self.y_values = data_series['y_values']
            self.x_pre = data_series['x_pre']
            self.x_post = data_series['x_post']

            # create the plot ---
            self.create_plot()
            # self.invert_y_axis()  # invert y axis 5% of the times

            # error bars ---
            self.plot_error_bars()

            # limits ---
            self.set_axis_limits()

            # spine ---
            self.configure_spine()

            # ticks ---
            self.configure_ticks()

            # gridlines ---
            self.configure_gridlines()

            # titles --
            self.configure_titles()

            # legend ---
            # self.configure_legend()

            # stats ---
            self.add_stat_info()
            self.add_confidence_band()
            self.add_random_equation()
            self.add_random_text()
            self.annotate_max_min()
            self.fill_vertical_region()
            self.annotate_each_point()
            self.add_drop_line()
            self.add_shape()
            self.highlight_horizontal_region()
            self.add_inset()
            self.add_inset_pie()
            self.add_inset_zoom()
            self.add_slope_indicator()
            self.add_shaded_relief()

            # --- SAVING ----------------------------------------------------------#
            save_path = os.path.join(self.cfg.output.image_dir, f'{graph_id}.jpg')

            if random.random() >= 0.75:
                self.fig.tight_layout()

                # Save the figure to a memory buffer in RGBA format
                buf = io.BytesIO()
                self.fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)

                # Load the image from the buffer using PIL
                img = Image.open(buf).convert('RGB')

                # add textures ---
                if self.texture_files is not None:
                    if random.random() <= 0.25:
                        texture = Image.open(random.choice(self.texture_files)).convert('RGB').resize(img.size)
                        img = Image.blend(img, texture, alpha=random.uniform(0.05, 0.15))

                if random.random() <= 0.1:
                    # Save the image as a Grayscale
                    if self.debug:
                        print('converting to grayscale')
                    img = img.convert('L')

                # graph_id = f'syn_line_{generate_random_string()}'

                # plt.savefig(save_path, bbox_inches='tight')
                img.save(save_path)
                buf.close()

                # plt.show()
                plt.close(self.fig)
                plt.close('all')
            else:
                self.fig.savefig(save_path, format='jpg', bbox_inches='tight')
                plt.close(self.fig)
                plt.close('all')

        except Exception as e:
            plt.close(self.fig)
            traceback.print_exc()
