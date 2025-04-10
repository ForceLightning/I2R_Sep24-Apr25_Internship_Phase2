# A*STAR/I2R Sep 2024 - Jun 2025 Internship (Phase 2)
An in-development repository for Christopher Kok's internship at A*STAR/I²R from Sep 2024 to Jun 2025.

# Introduction
This codebase is heavily reliant on code from [phase 1](https://github.com/ForceLightning/I2R_Sep24_Apr25_Internship/). Note that some changes to package management (using `uv` instead of `pipenv`) was implemented. A [dockerfile](./Dockerfile) is provided to build OpenCV-python from source as a dependency for Optical Flow-based methods.

# Requirements
- A CUDA-supported GPU.
- [CUDA Toolkit ≥ 12](https://developer.nvidia.com/cuda-downloads).
- Python ≥ 3.12.
- [`uv`](https://github.com/astral-sh/uv) for package management.

## Directory Structure
```sh
.
├── .python-version
├── .isort.cfg
├── Dockerfile
├── README.md
├── pyproject.toml
├── pyrightconfig.json
├── pytest.ini
├── requirements.txt
├── uv.lock
├── checkpoints                                     # A symlink may be used for this.
│   ├── cine
│   ├── lge
│   ├── residual-attention
│   ├── tscsenet
│   ├── sota
│   ├── two-plus-one
│   ├── two-stream
│   ├── urr-residual-attention
│   └── {others}
├── configs
│   └── *.yaml
├── data                                            # A symlink may be used for this.
│   ├── Indices                                     #   Used for splitting the train/val dataset.
│   ├── test                                        #   Test dataset.
│   │   ├── Cine                                    #       CINE image data
│   │   ├── LGE                                     #       LGE image data
│   │   └── masks                                   #       Annotated masks
│   └── train_val                                   #   Train/Val dataset.
│       ├── Cine                                    #       CINE image data
│       ├── LGE                                     #       LGE image data
│       └── masks                                   #       Annotated masks
├── dataset
├── docs
│   ├── Makefile
│   ├── build
│   ├── make.bat
│   └── source
│       ├── conf.py
│       └── index.rst
├── pretrained_models
│   ├── distance_measures_regressor.pth             # Used for Vivim SOTA model.
│   └── PNSPlus.pth                                 # May or may not be used for PNS+ SOTA model.
├── pyrightconfig.json
├── pytest.ini
├── requirements*.txt
├── src
└── thirdparty
    ├── TransUNet
    ├── VPS
    ├── fla_net
    └── vivim

```

# Installation
Clone the repository with:
```sh
git clone --recursive https://github.com/ForceLightning/I2R_Sep24-Apr25_Internship_Phase2.git
```
> [!NOTE]
> Ensure that the `--recursive` flag is set if third-party modules are needed.

Install the dependencies from `pyproject.toml` or `requirements.txt` or `uv.lock`. Note that the CUDA version used is 12.4, modify as necessary.

Optional dependencies can be installed with `uv`:
```sh
uv sync --all-groups
```

## CUDA-accelerated Optical Flow support
For OpenCV with CUDA support, look at the Dockerfile to see how it may be built, or see [OpenCV-Python](https://github.com/opencv/opencv-python?tab=readme-ov-file#manual-builds) manual build documentation for detailed instructions.

For reference, the build command used in this project was:
```sh
ENABLE_HEADLESS=1 ENABLE_CONTRIB=1 CMAKE_ARGS="-DCMAKE_C_COMPILER=/usr/bin/gcc -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_CUBLAS=ON -DWITH_MKL=ON -DMKL_USE_MULTITHREAD=ON -DPYTHON3_NUMPY_INCLUDE_DIRS=<PATH_TO_VIRTUALENV>/lib/python3.12/site-packages/numpy/_core/include/ -DWITH_GTK=ON -DWITH_OPENGL=ON" MAKEFLAGS="-j 16" uv build --wheel ../opencv-python
```
Set the path to the virutal env to find numpy header files and the number of discrete cpu cores for faster build times.

> [!NOTE]
> On Windows Subsystem for Linux (WSL) environments, ensure that the path to WSL libraries (after installing CUDA toolkit drivers and CUDNN libraries are in the `$PATH` and `$LD_LIBRARY_PATH` environment variables. This may be set in `~/.bashrc` or `~/.zshrc` configurations.
> ```sh
> export PATH="/usr/lib/wsl/lib:$PATH"
> export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
> ```

# Usage
Some default configurations are included in the `./configs/` directory.

> [!NOTE]
> Ensure that the environment variable `PYTHONPATH` is set to `./src/`.
> This can be done with:
> ```sh
> export PYTHONPATH="src/"
> ```
> Alternatively, set it in a `.env` file if using pipenv, or inside the `activate` script in the virtualenv `bin` directory.

## Description of config modules
Documentation WIP

## Useful CLI arguments
- `--version`: Sets the name of the experiment for logging and model checkpointing use cases.
- `--data.num_workers`: Set this to a value above 0 for dataloader workers to spawn. Defaults to 8.
- `--data.batch_size`: Set this to a value that will allow the model + data to fit in GPU memory.
- `--model.num_frames`: If the computational or memory complexity of the process overwhelms available compute, set this to a multiple of 5. Defaults to 5.
- `--model.weights_from_ckpt_path`: Set this to the path of a prior checkpoint to load the model's weights. Note that the config must be the same.
- `--model.encoder_name` and `--model.encoder_weights`: This sets the backbone architecture of the U-Net model and the desired weights for those available in the [SMP module](https://github.com/qubvel-org/segmentation_models.pytorch).
- `--data.augment`: This sets the dataset to augment images when loading them to the model during training only.

## Other modes (validate/test/predict)
With similar arguments as above, use the appropriate config mode `training.yaml`/`testing.yaml`/`predict.yaml` with the associated CLI subcommand:
```sh
python -m $MODULE validate ... --config testing.yaml
python -m $MODULE test ... --config testing.yaml
python -m $MODULE predict ... --config predict.yaml
```
