## benetech-making-graphs-accessible
## score at 5th position is achieved.
![benetech-submission](https://github.com/bishnarender/benetech-making-graphs-accessible/assets/49610834/f5734434-9029-4074-b57f-ddc77f1904de)

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
![graphs](https://github.com/bishnarender/benetech-making-graphs-accessible/assets/49610834/d2107627-4d52-4f92-9051-e2b819151eeb)

[Link](https://www.kaggle.com/datasets/narender129/mga-processed-deps) to processed images and annotations for synthetic data generation.

Generation code requires fonts such as FONT_FAMILY = ['DejaVu Sans', 'Arial', 'Times New Roman', 'Courier New', 'Helvetica', 'Verdana', 'Trebuchet MS', 'Palatino Linotype', 'Georgia', 'MesloLGS NF', 'Lucida Grande',
] are pre-installed. Copy ttf files of these to matplotlib location: "miniconda3/envs/base_4/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf/" and clear previous cache "rm ~/.cache/matplotlib -rf".

How is the mapping between label text and its corresponding coordinates, is the basis of a plot. As long as the minimum and maximum values are correct, the other results are not affected at all even if they are wrong, but conversely, if the minimum and maximum values are wrong, the accuracy will decrease even if the other results are correct.

### MGA (Making Graphs Accessible) model. 
-----
![matcha](https://github.com/bishnarender/benetech-making-graphs-accessible/assets/49610834/c814984b-7301-460b-b67d-aefed77b174a)

Three types of data have been trained over this model. First one contained all types of graphs and maximum possible examples. Also, this one was the base checkpoint for the next two types. Second one contained only scatter graphs. Third one contained graphs other than scatter.

Model input is simply the plot image itself (in the form of flattened patches) without any prompts.

Only decoder section of "Pix2Struct transformer model" has been used. Decoder models involve a pretraining task (called causal language modeling) where the model reads the texts in order and has to predict the next word. It’s usually done by reading the whole sentence with a mask to hide future tokens at a certain timestep.

Number of tokens in tokenizer: 50359 i.e., our vocabulary size.

Sequence length for flattened_patches is fixed and equal to "number of patches (~ 2048)". The shape is as ['flattened_patches'].shape => torch.Size([BS, 2048, 770]). The shape is derived by concatenating torch.Size([BS, 2048, 1]), torch.Size([BS, 2048, 1]) and torch.Size([BS, 2048, 768]). The first tensor torch.Size([BS, 2048, 1]) describes row_ids (indices of rows) repeated as many times as columns have. The second tensor torch.Size([BS, 2048, 1]) describes col_ids (indices of columns) repeated as many times as rows have. The last tensor torch.Size([BS, 2048, 768]) describes [BS, number of patches, channels*patch_size<sup>2</sup>].

Sequence length for texts is not fixed and changes according to contents. Maximum sequence length for texts is limited to 1024. The following is one of the texts (in the template form) as an example.
<code>
[<c_start>][<lines>][<c_end>][<p_start>]14|14[<p_end>][<x_start>]1918|1928|1938|1948|1958|1968|1978|1988|1998|2008|2018|2028|2038|2048[<x_end>][<y_start>]-4.51e+03|-3.10e+03|-1.99e+03|-1.16e+03|-5.44e+02|-1.19e+02|8.83e+01|2.63e+02|3.52e+02|2.55e+02|2.99e+02|3.73e+02|3.50e+02|4.58e+02[<y_end>]</s> 
</code>
As you can see numeric values have been cast into scientific notation using. <code>val = "{:.2e}".format(float(val))</code>

<b>Prediction: </b> By decoding the labels to text string, chart type and data series have been extracted.

#### Q. What is MatCha?
-----
<b>MatCha,</b> which stands for math and charts, is a pixels-to-text foundation model (a pre-trained model with built-in <b>inductive biases</b> that can be fine-tuned for multiple applications) trained on two complementary tasks: (a) chart de-rendering and (b) math reasoning. In chart de-rendering, given a plot or chart, the image-to-text model is required to generate its underlying data <b>table</b> or the <b>code</b> used to render it. For math reasoning pre-training, we pick textual numerical reasoning datasets and render the input into images, which the image-to-text model needs to decode for answers.

MatCha is a model that is trained using Pix2Struct architecture. MatCha use Pix2Struct as the base model and further pretrain it with chart derendering and math reasoning tasks.

<b>google/matcha-base:</b> the base MatCha model, used to fine-tune MatCha on downstream tasks.

#### Q. What is Pix2Struct?
-----
<b>Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding</b>. Pix2Struct, a pretrained image-to-text model for purely visual language understanding, which can be finetuned on tasks containing visually-situated language.
Visually-situated language is ubiquitous— sources range from textbooks with diagrams to web pages with images and tables, to mobile apps with buttons and forms.

Pix2Struct is an image-encoder-text-decoder based on ViT. ViT stands for Vision Transformer, a deep learning model proposed in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale”.

Pix2Struct has been fine-tuned on a variety of tasks and datasets, ranging from image captioning, visual question answering (VQA) over different inputs (books, charts, science diagrams), captioning UI components etc.

Pix2Struct preserves the aspect-ratio. The standard ViT scales the input images to a predefined resolution while Pix2Struct scales to a variable resolution. In order for the model to handle variable resolutions unambiguously, we use 2-dimensional absolute positional embeddings for the input patches.

![pix2struct](https://github.com/bishnarender/benetech-making-graphs-accessible/assets/49610834/97d85684-8b93-418a-bfdd-2585722e97e4)
[Image Reference](https://arxiv.org/pdf/2210.03347.pdf)

#### Q. What is ViT?
-----
<b>What ViT do?</b> Split an image into fixed-size small patches, flattens them, linearly embed each of them, add position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder. Image patches are treated the same way as tokens (words) in an NLP application.

The number of patches is decided via N = H*W / P<sup>2</sup>. Where H = height of image, W = width of image, and P = patch_size (usually 16 pixels). The standard Transformer receives as input a 1D sequence of token embeddings. To handle  images, we reshape the image x ∈ R<sup>H×W×C</sup> into a sequence of flattened patches x<sub>p</sub> ∈ R <sup>N×(P<sup>2</sup>·C)</sup> , where (H, W) is the resolution of the original image, C is the number of channels, (P, P) is the resolution of each image patch, and N = HW/P<sup>2</sup> is the resulting number of patches, which also serves as the effective input sequence length for the Transformer. The Transformer uses constant latent vector size D through all of its layers, so ViT flatten the patches and map to D dimensions with a trainable linear projection. We refer to the output of this projection as the patch embeddings. 

So, an image of size (640, 512, 3) is transformed into a sequence (N, P <sup>2</sup>·C) i.e., to sequence (1280, 768). P = patch_size is usually 16 pixels, N=1280 assumed. A [CLS] token is added to serve as representation of an entire image, which can be used for classification. The authors also add absolute position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder.

Position embeddings are added to the patch embeddings to retain positional information. ViT uses standard learnable 1D position embeddings, since it has not observed significant performance gains from using more advanced 2D-aware position embeddings.

![vision_transformer](https://github.com/bishnarender/benetech-making-graphs-accessible/assets/49610834/23eccec7-4a98-4a0f-823d-9fdddd5c6c0a)
[Image Reference](https://arxiv.org/pdf/2010.11929.pdf)

#### Q. What is AWP (Adverserial Weight Perturbation)?
-----
Adversarial examples are inputs to machine learning models that an attacker has intentionally designed to cause the model to make a mistake; they’re like optical illusions for machines.

AWP addresses the vulnerability of neural networks to adversarial attacks by perturbing the model's weights during training. The goal is to make the model more robust by adding a small amount of noise to the weights, which helps the model generalize better to unseen examples, including adversarial examples.

During training, AWP introduces random perturbations to the weights of the neural network. These perturbations can be additive noise or multiplicative factors applied to the original weights.

By adding noise to the weights, the model learns to be more robust to small changes in the input data. This helps prevent the model from relying too heavily on specific weight configurations, making it less susceptible to adversarial attacks.

The noise added to the weights should be carefully controlled to strike a balance between robustness and preserving the model's accuracy on clean data. If the perturbations are too large, it may lead to a drop in accuracy on clean data.

#### Q. What is EMA (Exponential Moving average)?
-----
Moving average is a smoothing technique that is commonly used for reducing the noise and fluctuation from time-series data.

In large neural networks, parameters are updated using certain mini-batch (including gradient accumulation steps) instead of using the entire dataset. This actually introduced the noise to the training. This training noise has both advantages and disadvantages.

Exponential moving average is a neural network training trick that sometimes improves the model accuracy. Exponential moving average (EMA) computes the weighted mean of all the previous data points and the weights decay exponentially.

The smoothing technique, such as EMA in particular, can be used to reduce the optimized parameter fluctuation noise and the parameter EMAs are more likely to be closed to a local minimum. Concretely, the optimized parameters after each update step are Θ<sub>1</sub>, Θ<sub>2</sub>, ⋯, Θ<sub>t</sub>, ⋯, Θ<sub>n</sub>, respectively. It’s just like a sequence of time-series data that has noise. Therefore, EMA can improve model generalization.

![ema](https://github.com/bishnarender/benetech-making-graphs-accessible/assets/49610834/28f25bb4-d504-4ea9-89e6-d97c53841a10)

where α∈(0,1].
