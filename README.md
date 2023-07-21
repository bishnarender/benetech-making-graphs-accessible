## benetech-making-graphs-accessible
## score at 5th position is achieved.


### Start 
-----
For better understanding of project, read the files in the following order:
1. eda.ipynb 
2. create_folds.ipynb
3. gen/run_gen_vbar.ipynb
4. synthetic_generation.ipynb
5. train_all.ipynb
6. benetech-submission.ipynb

### About
-----
Certain external data sources has been used:<br>
1. [TabEL: Entity Linking in WebTables.](http://websail-fe.cs.northwestern.edu/TabEL/)
2. ICDAR.
3. Self-mixup: created categorical series using [wikipedia glossary pages](https://www.kaggle.com/code/conjuring92/w03-stem-glossary/notebook) in STEM (Science, Technology, Engineering, and Mathematics) domain. created numerical series from random function generators.

TabLE and Self-mixup been used in the ratio 0.25:0.75.
<code>
generator = random.choices([wiki_generator, synthetic_generator], weights=[0.25, 0.75], k=1,)[0]
</code>

For categorical "data-series" in wiki-bank, only those are selected which do not have character's "Unicode code point" greater than 127. 

All the plots in synthetic have been generated using matplotlib.

[Link](https://www.kaggle.com/datasets/narender129/mga-processed-deps) to processed images and annotations for synthetic data generation.

Generation code requires fonts such as FONT_FAMILY = ['DejaVu Sans', 'Arial', 'Times New Roman', 'Courier New', 'Helvetica', 'Verdana', 'Trebuchet MS', 'Palatino Linotype', 'Georgia', 'MesloLGS NF', 'Lucida Grande',
] are pre-installed. Copy ttf files of these to matplotlib location: "miniconda3/envs/base_4/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/" and clear previous cache "rm ~/.cache/matplotlib -rf".

How is the mapping between label text and its corresponding coordinates, is the basis of a plot. As long as the minimum and maximum values are correct, the other results are not affected at all even if they are wrong, but conversely, if the minimum and maximum values are wrong, the accuracy will decrease even if the other results are correct.

### MGA (Making Graphs Accessible) model. 
-----
Three types of data have been trained over this model. First one contained all types of graphs and maximum possible examples. Also, this one was the base checkpoint for the next two types. Second one contained only scatter graphs. Third one contained graphs other than scatter.

Model input is simply the plot image itself (in the form of flattened patches) without any prompts.

Only decoder section of "Pix2Struct transformer model" has been used. Decoder models involve a pretraining task (called causal language modeling) where the model reads the texts in order and has to predict the next word. It’s usually done by reading the whole sentence with a mask to hide future tokens at a certain timestep.

Number of tokens in tokenizer: 50359 i.e., our vocabulary size.

Sequence length for flattened_patches is fixed and equal to "number of patches (~ 2048)". The shape is as ['flattened_patches'].shape => torch.Size([BS, 2048, 770]). The shape is derived by concatenating torch.Size([BS, 2048, 1]), torch.Size([BS, 2048, 1]) and torch.Size([BS, 2048, 768]). The first tensor torch.Size([BS, 2048, 1]) describes row_ids (indices of rows) repeated as many times as columns have. The second tensor torch.Size([BS, 2048, 1]) describes col_ids (indices of columns) repeated as many times as rows have. The last tensor torch.Size([BS, 2048, 768]) describes [BS, number of patches, patch_size<sup>2</sup>*channels].

Sequence length for texts is not fixed and changes according to contents. Maximum sequence length for texts is limited to 1024. The following is one of the texts (in the template form) as an example.
<code>
[<c_start>][<lines>][<c_end>][<p_start>]14|14[<p_end>][<x_start>]1918|1928|1938|1948|1958|1968|1978|1988|1998|2008|2018|2028|2038|2048[<x_end>][<y_start>]-4.51e+03|-3.10e+03|-1.99e+03|-1.16e+03|-5.44e+02|-1.19e+02|8.83e+01|2.63e+02|3.52e+02|2.55e+02|2.99e+02|3.73e+02|3.50e+02|4.58e+02[<y_end>]</s> 
</code>
As you can see numeric values have been cast into scientific notation using.



<code>val = "{:.2e}".format(float(val))</code>


<b>Prediction: </b> By decoding the labels to text string, chart type and data series have been extracted.
