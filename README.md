# Unmask Me - INRIA Project

## Code style

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
[![GitHub Super-Linter](https://github.com/Arthemide/UnmaskMe/workflows/Lint%20Code%20Base/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![js-standard-style](https://img.shields.io/badge/code%20style-standard-brightgreen.svg?style=flat)](https://github.com/feross/standard)
[![js-standard-style](https://img.shields.io/badge/deployed-heroku-blue.svg)](https://stormy-reaches-60483.herokuapp.com/)
[![js-standard-style](https://img.shields.io/badge/deployed%20version-1.0.0-green.svg)](https://stormy-reaches-60483.herokuapp.com/)

## Demo

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Live Demo](https://github.com/Arthemide/UnmaskMe/blob/add-demo/images/demo.gif)

## Features

- [x] Mask detection
- [x] Mask segmentation
- [x] CCGan to predict face
- [x] YOLO for mask detection

## Future Improvements

- [] Better mask segmentation
- [] StyleGan implementation

## ♻️&nbsp; Environment

In order to implement this project, our team exploited Google Colab/Kaggle’s resources. My first experimentations of the pre-processing steps were built on my laptop since they were not computationally expensive, but the model got trained on Colab/Kaggle using GPU.

## 🔑&nbsp; Prerequisites

All the dependencies and required libraries are included in the file [requirements.txt](https://github.com/Arthemide/UnmaskMe/tree/dev/requirements.txt)

## 🚀&nbsp; Installation

1. Clone the repository

    ```bash
    git clone https://github.com/Arthemide/UnmaskMe.git
    ```

2. Change your directory to the cloned repository

    ```bash
    cd UnmaskMe
    ```

3. Create a Python virtual environment named 'unmaskMe' and activate it

    ```bash
    pip3 install virtualenv
    python3 -m venv .unmaskme
    ```

    ```bash
    source .unmaskMe/bin/activate
    ```

4. Now, run the following command in your Terminal/Command Prompt to install the libraries required

    ```bash
    pip3 install -r requirements.txt
    ```

## 🧑🏻‍💻&nbsp; Working

- To detect face masks in an image type the following command:

```bash
python3 unmask_me_image.py --image images/pic1.jpg
```

- To detect face masks in real-time video streams type the following command:

```bash
python3 unmask_me_real_time.py
```

### Streamlit app

UnmaskMe webapp using Pytorch & Streamlit

```bash
python3 -m streamlit run app.py
```

## Link to the sub README's

- [Mask detection](https://github.com/Arthemide/UnmaskMe/blob/dev/mask_detection/README.md)
- [Mask segmentation](https://github.com/Arthemide/UnmaskMe/blob/dev/mask_segmentation/README.md)
- [CCGan](https://github.com/Arthemide/UnmaskMe/blob/dev/ccgan/README.md)

<!-- ## 📖&nbsp; References

### Mask detection

### Mask segmentation

### CcgaN -->

## 🙋🏻‍♂️&nbsp; Authors

- Marine Charra
- Nicolas Cotoni
- Amaury Delprat
- Adrien Duot
- Gireg Roussel

## ✍🏼&nbsp; Citation

If you find this repository useful, please use following citation

```bash
@misc{unmaskme_project_2021,
title={Unmask me project via CCGan},
author={Marine Charra, Nicolas Cotoni, Amaury Delprat, Adrien Duot and Gireg Roussel},
year={2021},
}
```

## 📚 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.
