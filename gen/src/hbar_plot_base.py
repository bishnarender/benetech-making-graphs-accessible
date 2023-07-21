import io
import os
import random
import string
import sys
import textwrap
import traceback
from copy import deepcopy
from itertools import cycle

import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms
import numpy as np
import seaborn as sns
from faker import Faker
from matplotlib.ticker import LogLocator, MaxNLocator
from PIL import Image

fake = Faker()

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

# --
TEXT_PRE_POST = [
    '*', 'o', 'v', '^', '+', 'a', 'b', 'c',
    'aa', 'a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)', 'i)', 'j)', 'k)', 'l)', 'm)', 'n)',
    '(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)',
]


def add_legend_name(title_dict):
    p = random.random()

    if p <= 0.3:
        return title_dict['y_title']
    elif p <= 0.6:
        return random.choice(RANDOM_LABELS)
    else:
        return generate_series_name()


def is_nan(val):
    return val != val


def generate_annotations(length=1):
    p = random.random()

    chars = string.ascii_lowercase
    digits = string.digits

    if p >= 0.5:
        anno = "".join(random.choices(chars, k=length))
    else:
        anno = "".join(random.choices(digits, k=length))
    return anno


def is_log_scale_feasible(input_list):
    input_list = deepcopy(input_list)
    min_value = min(input_list)
    range_value = max(input_list) - min(input_list)
    if (min_value > 0) & (range_value >= 1e4):
        return True
    return False


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


# ------


def parse_example(the_example):
    # get underlying data ---
    plot_title = the_example['plot_title']

    x_title = the_example['x_title']
    y_title = the_example['y_title']

    x_values = the_example['x_series']
    y_values = the_example['y_series']

    titles = {
        'x_title': x_title,
        'y_title': y_title,
        'plot_title': plot_title,
    }

    data_series = {
        'x_values': x_values,
        'y_values': y_values,
    }

    return titles, data_series

# ---- params -----------#


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
]

HATCH_OPTIONS = ['/', '\\', 'x', 'o', 'O', '.', '*']

COLOR_OPTIONS = ['b', 'g', 'r', 'c', 'm', 'k']

FONT_WEIGHTS = [
    'normal', 'bold', 'light', 'ultralight', 'heavy', 'black', 'semibold', 'demibold'
]


def generate_random_params():
    """generate diverse set of params for vertical bar plots
    """
    theme = random.choices(
        ['dark', 'white', 'ticks'],
        weights=[0.05, 0.05, 0.9],
        k=1
    )[0]

    # theme = 'ticks'

    if random.random() >= 0.75:
        font_weight = random.choice(FONT_WEIGHTS)
    else:
        font_weight = 'normal'

    if random.random() >= 0.5:
        hatch = ''
    else:
        hatch = random.choice(HATCH_OPTIONS)

    color = random.choice(COLOR_OPTIONS)

    if random.random() >= 0.25:
        edgecolor = color
    else:
        edgecolor = random.choice(COLOR_OPTIONS)

    # rotations ---
    rotation_x = random.choices([0, 45, 90], weights=[0.85, 0.10, 0.05], k=1)[0]
    rotation_y = random.choices([0, 45, 90], weights=[0.85, 0.10, 0.05], k=1)[0]

    params = {
        'sns_theme': theme,
        'plt_style': random.choice(plt.style.available),

        'color': color,
        'edgecolor': edgecolor,
        'hatch': hatch,

        'font_family': random.choice(FONT_FAMILY),
        'font_size': random.randint(7, 12),
        'font_weight': font_weight,
        'font_color': random.choice(COLOR_OPTIONS),

        'span_width': random.randint(12, 24),  # (8, 16)

        'dpi': 100,
        'width': random.uniform(4, 8),
        'height': random.uniform(5, 8),

        'max_pixels_w': 800,
        'max_pixels_h': 1200,

        'xtick_label_rotation': rotation_x,
        'ytick_label_rotation': rotation_y,

        'bottom_spine': random.random() >= 0.25,
        'aux_spine': random.random() >= 0.75,

        'x_grid': random.random() >= 0.90,
        'y_grid': random.random() >= 0.90,
        'x_minor_grid': random.random() >= 0.95,
        'y_minor_grid': random.random() >= 0.95,

        'add_legend': random.random() >= 0.95,

    }

    return params


class PixelAdjusterH:
    def __init__(self, params):
        params = deepcopy(params)
        self.params = params

        # main design variables ---
        self.font_size = params['font_size']
        self.max_y_tick_chars = params['max_y_tick_chars']
        self.rotation = params['ytick_label_rotation']
        self.span_width = params['span_width']

        self.n_points = params['n_points']
        self.max_pixels = params['max_pixels_h']
        self.soft_ub = params['height'] * params['dpi']
        self.height = params['height']

        if self.n_points >= 8:
            if random.random() >= 0.80:
                self.rotation = random.uniform(0, 45)
            else:
                self.rotation = 0

            self.max_y_tick_chars = 8

        if params['mean_y_chars'] >= 8:
            if random.random() >= 0.80:
                self.rotation = random.uniform(0, 45)
            else:
                self.rotation = 0
            self.max_y_tick_chars = 8
            self.span_width = random.randint(20, 24)

        if self.n_points >= 12:
            self.font_size = random.randint(6, 10)
            self.height = max(6.0, self.height)

        # constants
        self.font_scale = 0.75
        self.offset_h = 2

        # flag to see if constraints are met
        self.stop = False

        # actions
        self.actions = ['decrease_font_size']

    def compute_pixels(self):
        pixels = (self.font_scale * self.font_size) * self.max_y_tick_chars * (self.n_points + self.offset_h)
        return pixels

    def decrease_font_size(self):
        while self.font_size >= 6:  # requires minimum
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
            req_height = self.compute_pixels() / self.params['dpi']
            max_height = self.params['max_pixels_h'] / self.params['dpi']
            self.height = min(req_height, max_height)

        # return adjusted params ---
        self.params['height'] = self.height
        self.params['font_size'] = self.font_size
        self.params['ytick_label_rotation'] = self.rotation
        self.params['span_width'] = self.span_width

        return self.params


def adjust_params(params):
    params = deepcopy(params)
    if params['plt_style'] == 'dark_background':
        params['font_color'] = 'w'
        params['color'] = 'w'

    params = PixelAdjusterH(params).get_adjusted_params()

    if 'bold' in params['font_weight'].lower():
        params['font_size'] = min(8, params['font_size'])

    return params

#########################################################################################
# Main ---
#########################################################################################


class HorizontalBarPlot:
    def __init__(self, cfg, the_example, texture_files=None, debug=False):
        self.cfg = cfg
        self.example = deepcopy(the_example)
        self.params = generate_random_params()
        self.debug = debug
        self.texture_files = texture_files

        # create data dependent params ---
        self.params['n_points'] = len(the_example['y_series'])

        self.params['max_y_tick_chars'] = min(
            self.params['span_width'], max([len(str(y)) for y in the_example['y_series']])
        )
        self.params['mean_y_chars'] = np.mean([len(str(y)) for y in the_example['y_series']])

        # adjust params ---
        self.params = adjust_params(self.params)

        # configure style & create figure ---
        self.configure_style()
        self.fig, self.ax = self.get_figure_handles()

        # axis ---
        self.x_axis_type = 'standard'
        self.y_axis_type = 'standard'

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
        if random.random() <= 0.25:
            # print('drawing ebars')
            e_scaling = np.random.uniform(0.9, 1.1, len(self.y_values)) * np.absolute(self.x_values)
            x_errors = np.random.uniform(0.05, 0.1, len(self.y_values)) * e_scaling

            # clipping of error bar
            clip_val = (max(self.x_values) - min(self.x_values))*0.20
            x_errors = np.clip(x_errors, 0, clip_val)

            self.ax.errorbar(
                x=self.x_values,
                y=np.arange(0, len(self.y_values)),
                xerr=x_errors,
                fmt='none',  # 'none',
                ecolor=random.choice(COLOR_OPTIONS),
                elinewidth=random.uniform(0.8, 1.5),
                capsize=random.uniform(3, 5),
                barsabove=random.choice([True, False]),
                errorevery=random.randint(1, 2),
                alpha=random.uniform(0.25, 1.0),
            )

    # -----------------------------------------------------------------------------------#
    # ticks
    # -----------------------------------------------------------------------------------#

    def configure_ticks(self):
        # tick parameters---
        direction = random.choice(['in', 'out', 'inout'])

        x_top = (random.random() >= 0.6) & (self.params['aux_spine'])
        x_labeltop = (random.random() >= 0.8) & (x_top)

        y_right = (random.random() >= 0.6) & (self.params['aux_spine'])
        y_labelright = (random.random() >= 0.8) & (y_right) & (self.params['max_y_tick_chars'] <= 8)

        # set ticks for x axis
        num_y = len(self.y_values)
        y_tick_positions = np.arange(num_y)  # uniform x ticks
        self.ax.set_yticks(y_tick_positions)

        y_tick_labels = [
            "\n".join(textwrap.wrap(cat, width=self.params['span_width'])) for cat in self.y_values
        ]
        self.ax.set_yticklabels(y_tick_labels, minor=False)  # set x tick labels

        x_formatter = get_formatter(self.x_values)
        self.ax.xaxis.set_major_formatter(x_formatter)

        # set ticks for y axis
        x_range = max(self.x_values) - min(self.x_values)
        if (x_range > 1e5) & (self.x_axis_type == 'standard'):
            num_bins = random.randint(4, 6)
            tick_locator = MaxNLocator(nbins=num_bins)
            self.ax.yaxis.set_major_locator(tick_locator)

        if self.x_axis_type == 'log':
            tick_locator = LogLocator(base=self.x_log_base)
            self.ax.xaxis.set_major_locator(tick_locator)

        # if random.random() > 0.5:
        self.ax.minorticks_on()

        # set tick params
        self.ax.tick_params(
            axis='x',
            which='both',
            rotation=self.params['xtick_label_rotation'],
            direction=direction,
            top=x_top,  # ---
            labeltop=x_labeltop,
        )
        if random.random() >= 0.25:
            self.ax.xaxis.set_tick_params(which='minor', bottom=False, top=False)

        self.ax.tick_params(
            axis='y',
            which='both',
            rotation=self.params['ytick_label_rotation'],
            direction=direction,
            right=y_right,  # ---
            labelright=y_labelright,
        )

        if not is_numeric(self.y_values):
            self.ax.yaxis.set_tick_params(which='minor', left=False, right=False)

        elif random.random() >= 0.25:
            self.ax.yaxis.set_tick_params(which='minor', left=False, right=False)

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
        else:
            self.ax.yaxis.grid(visible=False, which='both')

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
            'size': self.params['font_size'] - random.randint(2, 4),
        }

        if self.params['add_legend']:
            if random.random() > 0.5:
                legend = self.ax.legend(prop=font_dict, loc='best', frameon=True)
                frame = legend.get_frame()         # frame.set_facecolor('red')
                frame.set_edgecolor(random.choice(COLOR_OPTIONS))
            else:
                self.ax.legend(prop=font_dict, loc='center left', bbox_to_anchor=(1, 0.5))

    # ------------------------------------------------------------------------------------#
    # titles
    # ------------------------------------------------------------------------------------#

    def configure_titles(self):
        # bounding box ---
        box_color = random.choice(['c', 'k', 'gray'])
        padding = random.uniform(0, 1)
        boxstyle = random.choice(['round', 'square'])

        if random.random() >= 0.80:
            bbox_props = {
                'facecolor': box_color,
                'alpha': random.uniform(0.25, 0.75),
                'pad': padding,
                'boxstyle': boxstyle
            }
        else:
            bbox_props = None

        # -------
        if self.params['font_color'] == 'y':  # no yellow font color
            self.params['font_color'] = 'k'

        font_dict = {
            'family': self.params['font_family'],
            'color': self.params['font_color'],
            'weight': self.params['font_weight'],
            'size': self.params['font_size'],
        }

        axes_font_dict = deepcopy(font_dict)
        axes_font_dict['size'] = self.params['font_size'] + random.randint(1, 2)

        x_location = 'center'
        y_location = 'center'

        if random.random() >= 0.85:
            x_location = random.choice(['left', 'right', 'center'])
            y_location = random.choice(['top', 'bottom', 'center'])

        if not self.params['aux_spine']:
            x_location = random.choice(['left', 'right'])
            y_location = random.choice(['top', 'bottom'])

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

        # figure title
        fig_width_inch = self.fig.get_figwidth()
        fig_dpi = self.fig.dpi
        fig_width_pixel = fig_width_inch * fig_dpi
        # We assume an average character width of 10 pixels
        char_width_pixel = 12
        num_chars = int(fig_width_pixel / char_width_pixel)

        # Set a long title and use textwrap to automatically wrap the text
        title = '\n'.join(textwrap.wrap(self.title_dict['plot_title'], num_chars))
        title_font_dict = deepcopy(font_dict)
        title_font_dict['size'] = self.params['font_size'] + random.randint(2, 4)

        title_loc = random.choice(['left', 'right', 'center'])

        if not self.params['aux_spine']:
            title_loc = random.choice(['left', 'right'])

        if self.params['aux_spine']:
            title_pad = 24
        else:
            title_pad = random.randint(6, 24)
            if random.random() >= 0.8:
                title_pad = - title_pad

        if random.random() >= 0.2:
            self.ax.set_title(title, fontdict=title_font_dict, loc=title_loc, pad=title_pad, bbox=bbox_props)

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

        if not self.params['bottom_spine']:
            self.ax.spines['bottom'].set_visible(False)
            active_spines.remove('bottom')

        # set linewidth ---
        linewidth = random.uniform(1.0, 3.0)
        for spine in active_spines:
            self.ax.spines[spine].set_linewidth(linewidth)

        # set color ---
        color = random.choice(['r', 'g', 'b', 'k'])
        if random.random() >= 0.75:
            color = 'k'
        for spine in active_spines:
            self.ax.spines[spine].set_color(color)

        if random.random() >= 0.975:
            offset = random.randint(10, 20)
            for spine in active_spines:
                self.ax.spines[spine].set_position(('outward', offset))

    # -----------------------------------------------------------------------------------#
    # axis transformation
    # -----------------------------------------------------------------------------------#

    def axis_transformation(self):
        activity = random.random() >= 0.75

        if (activity) & (is_log_scale_feasible(self.x_values)):
            self.x_axis_type = 'log'
            self.x_log_base = random.choice([2, 10])
            self.ax.set_xscale('log', base=self.x_log_base)
            print('log x axis')

    # -----------------------------------------------------------------------------------#
    # basic plot
    # -----------------------------------------------------------------------------------#

    def create_basic_plot(self):
        fill = random.random() >= 0.25

        if random.random() >= 0.5:
            linewidth = random.uniform(0.5, 1.0)
        else:
            factor = 2.0
            linewidth = np.random.rand(len(self.y_values))*factor
            linewidth = np.clip(linewidth, 0.25, 1.5)

        if (fill) & (random.random() >= 0.5):
            linewidth = 0.0

        color_list = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
        random.shuffle(color_list)
        color_cycle = cycle(color_list)

        prob = random.random()
        if prob <= 0.3:
            color = self.params['color']
        elif prob <= 0.5:
            color = [random.choice(['g', 'b', 'k']) for _ in range(len(self.x_values))]
        elif prob <= 0.7:
            color = [random.choice(['gray', 'k']) for _ in range(len(self.x_values))]
        else:
            color = [next(color_cycle) for _ in range(len(self.x_values))]

        # print(color)
        if (random.random() >= 0.9) & (len(self.x_values) <= 12):
            self.bin_width = [random.uniform(0.7, 0.9) for _ in range(len(self.x_values))]
        else:
            self.bin_width = random.uniform(0.7, 0.95)

        self.bars = self.ax.barh(
            y=np.arange(0, len(self.y_values)),
            width=self.x_values,
            height=self.bin_width,
            color=color,
            edgecolor=self.params['edgecolor'],
            hatch=self.params['hatch'],
            label=add_legend_name(self.title_dict),
            alpha=random.uniform(0.7, 1.0),
            linewidth=linewidth,
            fill=fill,
            zorder=random.randint(0, 2),
        )

        # trans = mtransforms.Affine2D().translate(-0.005, -0.005)
        # self.ax.barh(
        #     np.arange(0, len(self.y_values)),
        #     self.x_values,
        #     color="none",
        #     edgecolor="k",
        #     hatch="///",
        #     transform=trans+self.ax.transData,
        #     alpha=0.2,
        # )

        self.ax.invert_yaxis()

    # -----------------------------------------------------------------------------------#
    # special effects
    # -----------------------------------------------------------------------------------#

    def add_annotations(self):
        if random.random() >= 0.85:
            offset_percentage = random.uniform(0.05, 0.25)

            # Adding the text above the bars
            if len(self.y_values) <= 12:
                interval = random.randint(1, 3)
            else:
                interval = random.randint(3, 5)

            for idx, bar in enumerate(self.bars):
                x_val = bar.get_width()

                if x_val < 100:
                    text = f'{round(x_val+random.random(), 1)}'
                else:
                    text = "{:.1e}".format(x_val)

                prefix = ''
                if random.random() >= 0.25:
                    prefix = random.choice(TEXT_PRE_POST)
                suffix = ''
                if random.random() >= 0.25:
                    suffix = random.choice(TEXT_PRE_POST)

                p = random.random()
                if p <= 0.25:
                    text = f'{text}'
                elif p <= 0.5:
                    text = f'{prefix}'
                elif p <= 0.75:
                    text = f'{suffix}'
                else:
                    text = f'{prefix}-{suffix}'

                offset = x_val * offset_percentage
                if random.random() >= 0.5:
                    offset = -offset

                if idx % interval == 0:
                    self.ax.text(
                        x_val + offset,
                        bar.get_y() + bar.get_height() / 2,
                        text,
                        ha='left',
                        va='center'
                    )

    def add_numbers(self):
        if random.random() >= 0.95:
            offset_percentage = random.uniform(0.1, 0.2)

            # Adding the text above the bars
            if len(self.y_values) <= 12:
                interval = random.randint(1, 3)
            else:
                interval = random.randint(3, 5)

            for idx, bar in enumerate(self.bars):
                x_val = bar.get_width()

                text = f'{random.randint(1, 100)}%'
                if random.random() >= 0.5:
                    text = f'({text})'

                offset = - x_val * offset_percentage

                if idx % interval == 0:
                    self.ax.text(
                        x_val + offset,
                        bar.get_y() + bar.get_height() / 2,
                        text,
                        ha='center',
                        va='center'
                    )

    def add_random_text(self):
        if random.random() >= 0.95:
            if self.debug:
                print('adding random text')

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
                    rotation=random.choice([0, 90, 270]),
                )
            else:
                self.ax.text(
                    random.uniform(0.5, 0.7), random.uniform(0.1, 0.4), text,
                    fontsize=random.randint(8, 10),
                    color=random.choice(COLOR_OPTIONS),
                    alpha=random.uniform(0.2, 0.9),
                    transform=self.ax.transAxes,
                    rotation=random.choice([0, 90, 270]),
                    bbox=dict(
                        boxstyle='round,pad=0.5',
                        fc=f'{random.choice(COLOR_OPTIONS)}',
                        ec='k',
                        lw=random.uniform(0.9, 1.9),
                        alpha=random.uniform(0.2, 0.5),
                    )
                )

    def connect_by_line(self):
        if random.random() >= 0.98:
            if isinstance(self.bin_width, list):
                y_ = (np.cumsum(self.bin_width) - np.array(self.bin_width) / 2).tolist()
            else:
                y_ = self.y_values

            self.ax.plot(
                self.x_values,
                y_,
                color=random.choice(['r', 'g', 'b', 'y', 'c', 'm', 'k']),
                linewidth=random.uniform(0.8, 2.0),
                linestyle=random.choice(['-', '--', '-.', ':']),
            )

    def draw_line(self):
        if random.random() >= 0.975:
            color = random.choice(COLOR_OPTIONS)
            constant_y = random.uniform(0, len(self.y_values)-1)

            self.ax.axhline(
                constant_y,
                xmin=random.uniform(0.01, 0.25),
                xmax=random.uniform(0.75, 0.99),
                color=color,
                lw=random.uniform(0.5, 1.0),
                alpha=random.uniform(0.25, 0.75)
            )

        if random.random() >= 0.975:
            color = random.choice(COLOR_OPTIONS)
            constant_x = np.percentile(self.x_values, random.randint(20, 80))

            self.ax.axvline(
                constant_x,
                ymin=random.uniform(0.01, 0.25),
                ymax=random.uniform(0.75, 0.99),
                color=color,
                lw=random.uniform(0.5, 1.0),
                alpha=random.uniform(0.25, 0.75)
            )

    # -----------------------------------------------------------------------------------#
    # fancy bar caps
    # -----------------------------------------------------------------------------------#

    def add_caps(self):
        if random.random() >= 0.95:
            lw = random.uniform(0.5, 2.0)
            color = random.choice(COLOR_OPTIONS)

            for bar in self.bars:
                bar.set_clip_on(False)
                x = bar.get_width()
                y = bar.get_y()
                w = bar.get_height()
                self.ax.plot([x, x], [y, y+w], color=color, lw=lw, alpha=random.uniform(0.8, 1.0))

    # -----------------------------------------------------------------------------------#
    # tags
    # -----------------------------------------------------------------------------------#
    def add_tags(self):
        f = random.uniform(0.1, 0.2)

        if random.random() >= 0.95:
            for bar in self.bars:
                self.ax.text(bar.get_width()*f, bar.get_y() + bar.get_height()/2,  fake.word(), va="center")

    # -----------------------------------------------------------------------------------#
    # main api
    # -----------------------------------------------------------------------------------#

    def make_horizontal_bar_plot(self, graph_id):
        try:
            the_example = deepcopy(self.example)
            title_dict, data_series = parse_example(the_example)

            self.title_dict = title_dict
            self.x_values = data_series['x_values']
            self.y_values = data_series['y_values']

            # create the plot ---
            self.create_basic_plot()

            # add special effects ---
            self.connect_by_line()
            self.add_random_text()
            self.add_annotations()
            self.draw_line()
            self.add_numbers()
            self.plot_error_bars()
            # self.axis_transformation()

            if self.x_axis_type != 'log':
                self.add_caps()
                self.add_tags()

            self.configure_spine()
            self.configure_ticks()
            self.configure_gridlines()
            self.configure_legend()
            self.configure_titles()

            # sns.despine(left=True, bottom=True)

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
