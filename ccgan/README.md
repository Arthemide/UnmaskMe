# projet_inria

## Context-Conditional GAN

Sources for the ccgan : [here](https://github.com/eriklindernoren/PyTorch-GAN#context-conditional-gan)
From the paper [Semi-supervised learning with context-conditional generative adversarial networks](https://arxiv.org/pdf/1611.06430.pdf) by Emily Denton, Sam Gross, Rob Fergus.

### Dataset

You can download the dataset [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).<br>
In '**Align&Cropped Images**' Drive download '**Img/img_align_celeba.zip**' and extract it in the a new folder called **data** at the root of the project.

### Installation

<ol>
<li>Create a Python environment</li>
<ol>
<code>$ python -m venv env</code>
</ol>
<li>Activate the environment</li>
<ol>
<code>$ source env/bin/activate</code>
</ol>
<li>Install the requirements</li>
<ol>
<code>$ pip install -r requirements.txt</code>
</ol>
<ol>
<code>$ pip install -r requirements.txt</code>
</ol>
<ol>
<code>$ python ccgan/src/ccgan.py</code>
</ol>
</ol>