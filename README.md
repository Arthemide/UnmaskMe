# Unmask Me - INRIA Project

[![GitHub Super-Linter](https://github.com/Arthemide/UnmaskMe/workflows/Lint%20Code%20Base/badge.svg)](https://github.com/marketplace/actions/super-linter)

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
    pip install virtualenv
    virtualenv .unmaskMe
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
python3 unmask_me_image.py --image images/pic1.jpeg
```

- To detect face masks in real-time video streams type the following command:

```bash
python3 unmask_me_real_time.py 
```

### Streamlit app

UnmaskMe webapp using Pytorch & Streamlit

```bash
streamlit run app.py 
```

## Link to the sub README's

- [Mask detection](https://github.com/Arthemide/UnmaskMe/blob/dev/mask_detection/README.md)
- [Mask segmentation](https://github.com/Arthemide/UnmaskMe/blob/dev/mask_segmentation/README.md)
- [CCGan](https://github.com/Arthemide/UnmaskMe/blob/dev/ccgan/README.md)

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
