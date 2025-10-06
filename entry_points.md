The following is a list of commands for running my code.

## Create a virtual environment

    python -m venv venv

## Activate the virtual environment

    source venv/bin/activate

## Install python packages

    pip install -r requirements.txt

I ran into the issue described here: https://github.com/jax-ml/jax/issues/29042. The solution is to manually upgrade nvidia-cublas-cu12:

    pip install --upgrade nvidia-cublas-cu12==12.9.0.13

## Generate raw calibrated training data

    python src/generate_raw_data.py train
    python src/generate_raw_data.py test

## Generate initial estimates of depth and transit windows

    python src/generate_depth_estimation.py train
    python src/generate_depth_estimation.py test

## Generate limb darkening features

    python src/generate_limb_darkening.py train
    python src/generate_limb_darkening.py test

## Train models

    python src/train_models.py

## Generate submission

    python src/generate_submission.py test
