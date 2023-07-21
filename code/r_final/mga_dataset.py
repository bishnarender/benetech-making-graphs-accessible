import json
import os
from copy import deepcopy
from operator import itemgetter

import albumentations as A
import numpy as np
from PIL import Image
from tokenizers import AddedToken
from torch.utils.data import Dataset
from transformers import Pix2StructProcessor

# -- token map --#
TOKEN_MAP = {
    "line": "[<lines>]",
    "vertical_bar": "[<vertical_bar>]",
    "scatter": "[<scatter>]",
    "dot": "[<dot>]",
    "horizontal_bar": "[<horizontal_bar>]",
    "histogram": "[<histogram>]",

    "c_start": "[<c_start>]",
    "c_end": "[<c_end>]",
    "x_start": "[<x_start>]",
    "x_end": "[<x_end>]",
    "y_start": "[<y_start>]",
    "y_end": "[<y_end>]",

    "p_start": "[<p_start>]",
    "p_end": "[<p_end>]",

    "bos_token": "[<mga>]",
}

# -----


def fix_data_series(graph_id, x_series, y_series):
    if graph_id in [
        'a80688cb2101',
        'c48de2fcb4a4',
        '4566b5627dfc',
        '0df1338d2df7',
    ]:
        x_series = x_series[1:]
        y_series = y_series[1:]

    return x_series, y_series

# ref: https://www.kaggle.com/code/nbroad/donut-train-benetech


def is_nan(val):
    return val != val


def get_processor(cfg):
    """
    load the processor
    """
    processor_path = cfg.model.backbone_path
    print(f"loading processor from {processor_path}")
    
    # Constructs a PIX2STRUCT processor which wraps a BERT tokenizer and PIX2STRUCT image processor into a single processor.
    processor = Pix2StructProcessor.from_pretrained(processor_path)
    processor.image_processor.is_vqa = False
    processor.image_processor.patch_size = {
        "height": cfg.model.patch_size,# cfg.model.patch_size = 16
        "width": cfg.model.patch_size
    }

    # NEW TOKENS
    print("adding new tokens...")
    new_tokens = []
    for _, this_tok in TOKEN_MAP.items():
        new_tokens.append(this_tok)
    new_tokens = sorted(new_tokens)

    # new_tokens[0:4] => ['[<c_end>]', '[<c_start>]', '[<dot>]', '[<histogram>]']

    tokens_to_add = []
    for this_tok in new_tokens:
        # class tokenizers.AddedToken => represents a token that can be be added to a Tokenizer. 
        tokens_to_add.append(AddedToken(this_tok, lstrip=False, rstrip=False))

    # tokens_to_add[0:4] => [AddedToken("[<c_end>]", rstrip=False, lstrip=False, single_word=False, normalized=True), AddedToken("[<c_start>]", rstrip=False, lstrip=False, single_word=False, normalized=True), AddedToken("[<dot>]", rstrip=False, lstrip=False, single_word=False, normalized=True), AddedToken("[<histogram>]", rstrip=False, lstrip=False, single_word=False, normalized=True)]
    processor.tokenizer.add_tokens(tokens_to_add)

    return processor


class MGADataset(Dataset):
    """Dataset class for MGA dataset
    """

    def __init__(self, cfg, graph_ids, transform=None):

        self.cfg = cfg
        self.data_dir = cfg.competition_dataset.data_dir.rstrip("/")
        self.image_dir = os.path.join(self.data_dir, "train", "images")
        self.annotation_dir = os.path.join(self.data_dir, "train", "annotations")

        self.syn_data_dir = cfg.competition_dataset.syn_dir.rstrip("/")
        self.syn_image_dir = os.path.join(self.syn_data_dir, "images")
        self.syn_annotation_dir = os.path.join(self.syn_data_dir, "annotations")

        self.pl_data_dir = cfg.competition_dataset.pl_dir.rstrip("/")
        self.pl_image_dir = os.path.join(self.pl_data_dir, "images")
        self.pl_annotation_dir = os.path.join(self.pl_data_dir, "annotations")

        self.graph_ids = deepcopy(graph_ids)
        self.transform = transform

        # load processor
        self.load_processor()

    def load_processor(self):
        self.processor = get_processor(self.cfg)

    def load_image(self, graph_id):
        if ("syn_" in graph_id) | ("ext_" in graph_id):
            try:
                image_path = os.path.join(self.syn_image_dir, f"{graph_id}.jpg")
                image = Image.open(image_path)  # .convert('RGBA')
                image = image.convert('RGB')
            except Exception as e:
                image_path = os.path.join(self.syn_image_dir, f"{graph_id}_v0.jpg")
                image = Image.open(image_path)  # .convert('RGBA')
                image = image.convert('RGB')

        elif "pl_" in graph_id:
            image_path = os.path.join(self.pl_image_dir, f"{graph_id}.jpg")
            image = Image.open(image_path)  # .convert('RGBA')
            image = image.convert('RGB')

        else:
            image_path = os.path.join(self.image_dir, f"{graph_id}.jpg")
            image = Image.open(image_path)
            image = image.convert('RGB')

        return image

    def process_point(self, val, d_type, chart_type):
        """
        process the x/y point in a suitable format
        """
        # handling of numerical points ---
        if d_type == "numerical":
            if chart_type != "dot":
                val = "{:.2e}".format(float(val))
            else:
                val = str(int(float(val)))  # only counts for dot charts

        elif d_type == "categorical":
            val = val  # val.strip()

        else:
            raise TypeError

        return val

    def build_output(self, graph_id):
        if ("syn_" in graph_id) | ("ext_" in graph_id):
            annotation_path = os.path.join(self.syn_annotation_dir, f"{graph_id}.json")
        elif "pl_" in graph_id:
            annotation_path = os.path.join(self.pl_annotation_dir, f"{graph_id}.json")
        else:
            annotation_path = os.path.join(self.annotation_dir, f"{graph_id}.json")

        # read annotations ---
        with open(annotation_path, "r") as f:
            annotation = json.load(f)

        # store necessary data for labels ---
        chart_type = annotation['chart-type']
        chart_data = annotation['data-series']

        # sort chart data in case of scatter plot ---
        if chart_type == "scatter":
            chart_data = sorted(chart_data, key=itemgetter('x', 'y'))

        x_dtype = annotation['axes']['x-axis']['values-type']
        y_dtype = annotation['axes']['y-axis']['values-type']

        x_series = [d['x'] for d in chart_data if not is_nan(d['x'])]
        y_series = [d['y'] for d in chart_data if not is_nan(d['y'])]

        # detect histogram --
        if len(x_series) != len(y_series):
            chart_type = "histogram"

        x_series, y_series = fix_data_series(graph_id, x_series, y_series)

        num_x = len(x_series)
        num_y = len(y_series)

        # x_max = None
        # if x_dtype == "numerical":
        #     x_max = max([abs(float(x)) for x in x_series])

        # y_max = None
        # if y_dtype == "numerical":
        #     y_max = max([abs(float(y)) for y in y_series])

        x_series = [self.process_point(x, x_dtype, chart_type) for x in x_series]
        y_series = [self.process_point(y, y_dtype, chart_type) for y in y_series]

        c_string = TOKEN_MAP["c_start"] + TOKEN_MAP[chart_type] + TOKEN_MAP["c_end"]
        p_string = TOKEN_MAP["p_start"] + f"{num_x}|{num_y}" + TOKEN_MAP["p_end"]
        x_string = TOKEN_MAP["x_start"] + "|".join(x_series) + TOKEN_MAP["x_end"]
        y_string = TOKEN_MAP["y_start"] + "|".join(y_series) + TOKEN_MAP["y_end"]
        e_string = self.processor.tokenizer.eos_token

        text = f"{c_string}{p_string}{x_string}{y_string}{e_string}"        
        # text = f"{c_string}{x_string}{y_string}{e_string}"
        
        # text => 
        # [<c_start>][<scatter>][<c_end>][<p_start>]14|14[<p_end>][<x_start>]-4.00e+01|-3.89e+01|-3.39e+01|-3.33e+01|-3.24e+01|-3.21e+01|-2.87e+01|-2.82e+01|-2.52e+01|-2.48e+01|-2.37e+01|-2.35e+01|-2.33e+01|-2.30e+01[<x_end>][<y_start>]1.35e+00|-8.82e-01|5.51e-01|4.66e-01|2.10e+00|-2.60e-01|-3.62e-01|-1.45e+00|1.06e+00|3.53e-01|5.83e-01|-1.88e+00|-6.53e-01|-9.73e-01[<y_end>]</s>

        # text => 
        # [<c_start>][<lines>][<c_end>][<p_start>]14|14[<p_end>][<x_start>]1918|1928|1938|1948|1958|1968|1978|1988|1998|2008|2018|2028|2038|2048[<x_end>][<y_start>]-4.51e+03|-3.10e+03|-1.99e+03|-1.16e+03|-5.44e+02|-1.19e+02|8.83e+01|2.63e+02|3.52e+02|2.55e+02|2.99e+02|3.73e+02|3.50e+02|4.58e+02[<y_end>]</s>        

        return text, chart_type

    def __str__(self):
        string = 'MGA Dataset'
        string += f'\tlen = {len(self)}\n'
        return string

    def __len__(self):
        return len(self.graph_ids)

    def __getitem__(self, index):
        graph_id = self.graph_ids[index]
        image = self.load_image(graph_id)

        if self.transform is not None:
            image = np.array(image)
            image = self.transform(image=image)["image"]

        try:
            text, chart_type = self.build_output(graph_id)
        except Exception as e:
            print(f"Error in {graph_id}")
            print(e)  # e
            text, chart_type = 'error', 'error_chart'

        # image processor ---
        p_img = self.processor(
            images=image,
            # the maximum number of patches to extract from the image.
            # aspect-ratio preserving patches,
            max_patches=self.cfg.model.max_patches,# 2048
            add_special_tokens=True,
        )

        # process text
        p_txt = self.processor(
            text=text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.cfg.model.max_length,# 1024
        )
         
        r = {}
        r['id'] = graph_id
        r['chart_type'] = chart_type
        r['image'] = image
        r['text'] = text
        r['flattened_patches'] = p_img['flattened_patches']
        
        # We also pass attention_mask as additional input to the model, which makes sure that padding tokens of the inputs are ignored. This argument indicates to the model which tokens should be attended to, and which should not.  
        r['attention_mask'] = p_img['attention_mask']

        # The input ids are often the only required parameters to be passed to the model as input. They are token indices, numerical representations of tokens building the sequences that will be used as input by the model. The tokens are either words or subwords. 
        try:
            r['decoder_input_ids'] = p_txt['decoder_input_ids']
        except KeyError:
            r['decoder_input_ids'] = p_txt['input_ids']

        # decoder_attention_mask => generate a tensor that ignores pad tokens in decoder_input_ids.
        try:
            r['decoder_attention_mask'] = p_txt['decoder_attention_mask']
        except KeyError:
            r['decoder_attention_mask'] = p_txt['attention_mask']
            
        # r =>
        # {'id': 'syn_scatter_jeyosx8f', 'chart_type': 'scatter', 'image': array(...), 'text': '[<c_start>][<scatter>]...[<y_end>]</s>', 'flattened_patches': [array(...)], 'attention_mask': [array(...)], 'decoder_input_ids': [50345, 50353, ..., 50357, 1, 1], 'decoder_attention_mask': [1, 1, 1, ..., 1, 1, 1]}
        
        # r['image'].shape => (241, 395, 3)
        # r['flattened_patches'][0].shape => (2048, 770) => fix for every element.
        # r['attention_mask'][0].shape => (2048,) => fix for every element.
        
        return r


# # ---- transforms ----#

def create_train_transforms():
    """
    Returns transformations.

    Returns:
        albumentations transforms: transforms.
    """

    transforms = A.Compose(
        [
            A.OneOf(
                [
                    A.RandomToneCurve(scale=0.3),
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.1, 0.2),
                        contrast_limit=(-0.4, 0.5),
                        brightness_by_max=True,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=(-20, 20),
                        sat_shift_limit=(-30, 30),
                        val_shift_limit=(-20, 20)
                    )
                ],
                p=0.5,
            ),

            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3),
                    A.MedianBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=3),
                    A.GaussNoise(var_limit=(3.0, 9.0)),
                ],
                p=0.5,
            ),

            A.Downscale(always_apply=False, p=0.1, scale_min=0.90, scale_max=0.99),
        ],

        p=0.5,
    )
    return transforms
