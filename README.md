# projet_inria

## Context-Conditional GAN

Source for the ccgan : https://github.com/eriklindernoren/PyTorch-GAN

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
</ol>

### Run context encoder

<code>$ python ccgan/src/ccgan.py</code>