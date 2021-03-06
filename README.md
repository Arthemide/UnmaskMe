# Unmask Me - INRIA Project

## Code style

[![GitHub Super-Linter](https://github.com/Arthemide/UnmaskMe/workflows/Lint%20Code%20Base/badge.svg)](https://github.com/marketplace/actions/super-linter)
[![GitHub Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Demo

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Live Demo](https://github.com/Arthemide/UnmaskMe/blob/master/images/demo.gif)

## Features

- [x] Mask detection
- [x] Mask segmentation
- [x] CCGan to predict face
- [x] YOLO for mask detection

## Future Improvements

- [] Better mask segmentation
- [] StyleGan implementation

## β»οΈ&nbsp; Environment

In order to implement this project, our team exploited Google Colab/Kaggleβs resources. My first experimentations of the pre-processing steps were built on my laptop since they were not computationally expensive, but the model got trained on Colab/Kaggle using GPU.

## π&nbsp; Prerequisites

All the dependencies and required libraries are included in the file [requirements.txt](https://github.com/Arthemide/UnmaskMe/tree/dev/requirements.txt)

## π&nbsp; Installation

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

## π§π»βπ»&nbsp; Working

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

## Link to the sub README's and report

- [Mask detection](https://github.com/Arthemide/UnmaskMe/blob/master/mask_detection/README.md)
- [Mask segmentation](https://github.com/Arthemide/UnmaskMe/blob/master/mask_segmentation/README.md)
- [CCGan](https://github.com/Arthemide/UnmaskMe/blob/master/ccgan/README.md)
- [Report](https://github.com/Arthemide/UnmaskMe/blob/master/documentation/report.pdf)

<!-- ## π&nbsp; References

### Mask detection

### Mask segmentation

### CcgaN -->

## ππ»ββοΈ&nbsp; Authors

- Marine Charra
- Nicolas Cotoni
- Amaury Delprat
- Adrien Duot
- Gireg Roussel

## βπΌ&nbsp; Citation

If you find this repository useful, please use following citation

```bash
@misc{unmaskme_project_2021,
title={Unmask me project via CCGan},
author={Marine Charra, Nicolas Cotoni, Amaury Delprat, Adrien Duot and Gireg Roussel},
year={2021},
}
```

## π License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.
