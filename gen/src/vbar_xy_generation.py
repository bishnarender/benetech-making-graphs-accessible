
import json
import os
import random
import sys
from copy import deepcopy

import numpy as np
from scipy.special import factorial, gamma, zeta

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
    from categories import CATEGORIES
    from constants import YEAR_TITLES
    from function_generator import generate_y
    from generator_utils import (detect_year, generate_range,
                                 has_non_latin_chars, is_numeric)
    from metadata_generator import generate_thematic_metadata
except ImportError:
    raise ImportError('Importing failed.')


# ---------------------------------------------------------------------------------------#
# utils ---------------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------#

def update_wiki_series(
    x_series,
    y_series,
    max_char_allowed=148,
    max_char_per_element=32,
):
    # copy the series ---
    # mutable elements in a shallow copy point to the original elements but those in a deep copy don’t.
    x_series = deepcopy(x_series)
    y_series = deepcopy(y_series)

    x_series = [str(x) for x in x_series]
    
    # x_series => ['Communist Party of Czechoslovakia', 'Czech National Social Party', "Czechoslovak People's Party", 'Democratic Party', 'Czechoslovak Social Democracy', 'Communist Party of Slovakia', 'Freedom Party', 'Labour Party', 'Invalid/blank votes', 'Total']
    
    # x_series[:-1] => ['Communist Party of Czechoslovakia', 'Czech National Social Party', "Czechoslovak People's Party", 'Democratic Party', 'Czechoslovak Social Democracy', 'Communist Party of Slovakia', 'Freedom Party', 'Labour Party', 'Invalid/blank votes']
    
    # ignore total 50% of the times ---
    if 'total' in x_series[-1].lower():
        if random.random() >= 0.25:
            x_series = x_series[:-1]
            y_series = y_series[:-1]

    # tracker ---
    char_count = 0
    for idx, x_val in enumerate(x_series):
        char_count += min(len(x_val), max_char_per_element)
        if char_count >= max_char_allowed:
            break

    x_series = x_series[:idx + 1]
    y_series = y_series[:idx + 1]

    # normalization ---
    threshold = 5e5
    y_series = np.array(y_series)
    max_val = np.absolute(y_series).max()

    if max_val > threshold:
        # random.uniform(a, b) => return a random floating point number N such that a <= N <= b.
        y_series = y_series * (random.uniform(threshold*0.5, threshold*1.5) / max_val)
    if random.random() >= 0.10:
        y_series = np.absolute(y_series)
    y_series = list(y_series)

    return x_series, y_series

# ---------------------------------------------------------------------------------------#
# wiki generator ------------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------#


def generate_from_wiki(wiki_bank):
    #example = random.choice(wiki_bank)
    
    # example =>
#     [{'plot-title': '', 'series-name': 'Pos', 'data-type': 'numeric', 'data-series': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}, {'plot-title': '', 'series-name': 'Club', 'data-type': 'categorical', 'data-series': ['Tonbridge reserves', ...
    
    
    for example in wiki_bank:

        cat_series_list = []
        num_series_list = []
        plot_title = example[0]['plot-title']

        for field in example:
            # field =>
            #{.., 'data-series': [..., 'Rómovia', 'Movement of Czechoslovakian Understanding', "People's Democratic Party–Rally for the Republic", 'Free Bloc', 'Czechoslovakian Socialist Party', 'Czechoslovakian Democratic Forum', 'Invalid/blank votes', 'Total', 'Registered voters/turnout']}            
            
            if field['data-type'] == 'categorical':
                # check if its unique ---
                if len(set(field['data-series'])) == len(field['data-series']):
                    if not has_non_latin_chars(field['data-series']):
                        cat_series_list.append(field)
            else:
                num_series_list.append(field)                
        
        if (len(cat_series_list) > 0) & (len(num_series_list) > 0):
            # cat_series_list => [{'plot-title': '', 'series-name': 'Party', 'data-type': 'categorical', 'data-series': ['Communist Party of Czechoslovakia', 'Czech National Social Party', "Czechoslovak People's Party", 'Democratic Party', 'Czechoslovak Social Democracy', 'Communist Party of Slovakia', 'Freedom Party', 'Labour Party', 'Invalid/blank votes', 'Total']}]

            # num_series_list => [{'plot-title': '', 'series-name': 'Votes', 'data-type': 'numeric', 'data-series': [2205697, 1298980, 1111009, 999622, 855538, 489596, 60195, 50079, 59427, 7130143]}]
            
            xs = random.choice(cat_series_list)
            ys = random.choice(num_series_list)

            # prune series ---
            x_series = xs['data-series']
            y_series = ys['data-series']
            x_series, y_series = update_wiki_series(x_series, y_series)

            anno = {
                'plot_title': plot_title,

                'x_title': xs['series-name'],
                'y_title': ys['series-name'],

                'x_series': x_series,
                'y_series': y_series,
            }
            
            #break# my break
            yield anno
                    


# ---------------------------------------------------------------------------------------#
# synthetic generator -------------------------------------------------------------------#
# ---------------------------------------------------------------------------------------#
def generate_categories(
        n_points,
        stem_glossary,
        max_char_allowed=148,
        max_char_per_element=32,
        **kwargs
):
    p = random.random()
    if p <= 0.1:
        key = random.choice(list(CATEGORIES.keys()))
        kv = CATEGORIES[key]

    else:
        key = random.choice(list(stem_glossary.keys()))
        kv = stem_glossary[key]

    n = min(n_points, len(kv))
    x_cats = random.sample(kv, n)

    # tracker ---
    char_count = 0
    for idx, x_val in enumerate(x_cats):
        char_count += min(len(x_val), max_char_per_element)
        if char_count >= max_char_allowed:
            break

    x_cats = x_cats[:idx + 1]
    x_cats = list(set(x_cats))
    if len(x_cats) <= 1:
        x_cats = ['Category 1', 'Category 2']

    return x_cats


def generate_years(n_points, uniform=True, **kwargs):
    """generate a list of years
    """

    # start and end years ---
    start_year = random.randint(1900, 2020)
    end_year = random.randint(start_year + 20, 2100)

    default_interval = random.choices(
        [1, 2, 3, 4, 5, 10],
        weights=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
        k=1,
    )[0]

    # generate the years with a constant/variable interval
    ret = [start_year]

    for _ in range(n_points - 1):
        if uniform:
            interval = default_interval
        else:
            interval = random.randint(1, 10)
        next_year = ret[-1] + interval

        if next_year > end_year:
            break
        ret.append(next_year)

    return ret


def generate_integers(n_points, uniform=True, length_scale=25, **kwargs):
    """generate a list of integers
    """
    assert length_scale > 0

    p = random.random()
    if p <= 0.1:
        start = random.randint(0, 2)
    elif p <= 0.5:
        start = random.randint(0, n_points*length_scale)
    else:
        start = random.randint(-n_points*length_scale, n_points*length_scale)

    default_interval = random.randint(1, length_scale)

    # generate the years with a constant/variable interval
    points = [start]
    for _ in range(n_points - 1):
        prev = points[-1]
        if uniform:
            interval = default_interval
        else:
            interval = random.randint(1, length_scale)

        points.append(prev + interval)
    return points


def generate_floats(n_points, uniform=True, length_scale=25.0, **kwargs):
    """generate a list of floats    
    """
    assert length_scale > 0

    # number of decimal places
    n_d = 1  # random.choices([1, 2, 3], weights=[0.75, 0.20, 0.05], k=1)[0]

    p = random.random()
    if p <= 0.1:
        start = random.uniform(0, 2)
    elif p <= 0.5:
        start = random.uniform(0, n_points*length_scale)
    else:
        start = random.uniform(-n_points*length_scale, n_points*length_scale)

    default_interval = random.uniform(0.1, length_scale)

    points = [start]
    for _ in range(n_points - 1):
        prev = points[-1]
        if uniform:
            interval = default_interval
        else:
            interval = random.uniform(0, length_scale)

        points.append(prev + interval)
    
    # round(number, ndigits=None) => Return number rounded to ndigits precision after the decimal point. If ndigits is omitted or is None, it returns the nearest integer to its input.
    # round(2.675, 2) gives 2.67; round(0.5) and round(-0.5) are 0; round(1.5) is 2.
    points = [round(point, n_d) for point in points]

    return points


def generate_x(stem_glossary):
    """
    main function to generate x series
    """
    
    n_points = random.randint(4, 24)
    # n_points => 13
    
    func_list = [generate_categories, generate_range, generate_years, generate_integers, generate_floats]
    # random.choices(population, weights=None, *, cum_weights=None, k=1) => Return a k sized list of elements chosen from the population with replacement.
    func = random.choices(func_list, weights=[0.0, 0.1, 0.1, 0.1, 0.1], k=1)[0]    
    # func => <function generate_floats at 0x7f740118ed30>
    
    length_scale = random.randint(1, 10)    
    
    uniformity = random.random() > 0.90

    return func(n_points, uniform=uniformity, length_scale=length_scale, stem_glossary=stem_glossary)

# -----------------------------------------------------------------------------#
# --- generate y series -------------------------------------------------------#


def generate_y_values(x):
    """generate y series
    """
    y = generate_y(x)

    # force positive
    if random.random() >= 0.02:
        if random.random() >= 0.90:
            # .amin() => amin is an alias of min.
            y = np.array(y) - np.amin(y)
        else:
            y = np.absolute(y)

    return list(y)


# --- generate data -----------------------------------------------------------#


def generate_xy(stem_glossary):
    xs = generate_x(stem_glossary)
    ys = generate_y_values(xs)

    numeric_flag = is_numeric(xs)
    # histogram data
    if (numeric_flag) & (random.random() >= 0.0):  # histogram
        ys[-1] = np.nan  # nan value

    metadata = generate_thematic_metadata()
    # metadata =>{'xlabel': 'Execution Time (milliseconds)', 'ylabel': 'API Response Time (milliseconds)', 'title': 'JavaScript Security Best Practices'}
    
    plot_title = metadata['title']

    # cast x axis to string
    xs = [str(x) for x in xs]
    x_title = metadata['xlabel']
    if detect_year(xs):
        x_title = random.choice(YEAR_TITLES)
    y_title = metadata['ylabel']

    # -----
    to_return = {
        'plot_title': plot_title,

        'x_title': x_title,
        'y_title': y_title,

        'x_series': xs,
        'y_series': ys,
    }
    # ---
    
    # to_return =>
    # {'plot_title': 'Gravitational Force and Geocentric Distance of a Moon', 'x_title': 'Geocentric Distance (meters)', 'y_title': 'Escape Energy (joules)', 'x_series': ['-36.0', '-34.7', '-31.5', '-31.1', '-30.9', '-30.7', '-28.7', '-25.1', '-24.0', '-23.5'], 'y_series': [92.540519824646, 81.59038280675628, 70.81131673539949, 60.20152713533779, 49.759242804657056, 39.482715516512236, 29.370219724749468, 19.420052273292306, 9.630532109294869, nan]}
    
    return to_return


def generate_from_synthetic(stem_glossary):
    #yield generate_xy(stem_glossary)
    while True:# infinte loop stops when yield executed.
        yield generate_xy(stem_glossary)


# REQ:
# 1. generate x series of categorical data
# 2. generate histograms
# 3. unequal spacing
# 4. unequal width
# 5. more examples with many bars
# 6. lengthy category names

# scatter 45 done
