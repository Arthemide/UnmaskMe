# Unmask Me - INRIA Project

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

<!-- 1. Open terminal. Go into the cloned project directory and type the following command:
```
$ python3 train_mask_detector.py --dataset dataset
``` -->

- To detect face masks in an image type the following command:

```bash
python3 unmask_me_image.py --image images/pic1.jpeg
```

- To detect face masks in real-time video streams type the following command:

```bash
python3 unmask_me_realtime.py 
```

## Streamlit app

UnmaskMe webapp using Pytorch & Streamlit

command

```bash
streamlit run app.py 
```

[![GitHub Super-Linter](https://github.com/Arthemide/UnmaskMe/workflows/Lint%20Code%20Base/badge.svg)](https://github.com/marketplace/actions/super-linter)
