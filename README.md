# AI Final

### Quickdraw Classification Problem
**website:** https://www.kaggle.com/competitions/quickdraw-doodle-recognition/overview <br>
**dataset:** shuffled from train_simplified from https://www.kaggle.com/competitions/quickdraw-doodle-recognition/data.

---
## Training

### Train on Google Colab
- upload training data to google drive
- Run all cells in quickdraw.ipynb 

### Train at Local machine
### Prerequisites
#### Mac with apple silicon
- Download Conda env: https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
1. Install Miniforge3
```sh
    chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh
    sh ~/Downloads/Miniforge3-MacOSX-arm64.sh
    source ~/miniforge3/bin/activate
```
2. Create a Conda Environment
```sh
    conda create -n venv
```
3. Activate Conda Environment
```sh
    conda activate venv
```
4. Install the TensorFlow dependencies
```
    conda install -c apple tensorflow-deps
```
5. Install base Tensorflow
```sh
    python -m pip install tensorflow-macos
```
6. Install tensorflow-metal plugin
```
    python -m pip install tensorflow-metal
```

### Start Training
1. Open Jupyter Lab
```sh
    jupyter lab
```
2. Train
    - clone this github repository
    - Either run all cells in quickdraw.ipynb or run the main.py in branch `package`

## Submit to Kaggle
- **website**: https://www.kaggle.com/competitions/quickdraw-doodle-recognition/submit

## References
1. https://www.kaggle.com/code/gaborfodor/greyscale-mobilenet-lb-0-892
2. https://www.kaggle.com/code/kotarojp/first-step-for-submission-keras-resnet50
3. https://www.kaggle.com/competitions/quickdraw-doodle-recognition/discussion/70558
