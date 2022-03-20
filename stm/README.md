# NOTE: The below README is the original README from the [STM-cycle](https://github.com/lyxok1/STM-Training) code base, with the exception of the davis17eval submodule information and a few minor spelling changes.

# Training Script for Space Time Memory Network

This codebase implements training code for [Space Time Memory Network](http://openaccess.thecvf.com/content_ICCV_2019/html/Oh_Video_Object_Segmentation_Using_Space-Time_Memory_Networks_ICCV_2019_paper.html) with some [cyclic features](https://arxiv.org/abs/2010.12176).

<img src="demo.gif" alt="sample results" style="max-width:75%;">

## Requirements
### python packages
- torch
- python-opencv
- pillow
- yaml
- imgaug
- yacs
- progress
- [nvidia-dali](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html) (optional)

### Submodule: davis17eval
Clone https://github.com/davisvideochallenge/davis2017-evaluation as a submodule into the `stm` directory with name `davis17eval`.

### GPU support

- GPU Memory >= 12GB
- CUDA >= 10.0

## Data

See the doc [DATASET.md](./DATASET.md) for more details on data organization of our prepared dataset.

## Release
We provide pre-trained models with different backbone in our codebase, results are validated on DAVIS17-val with gradient correction.

| model |backbone|data backend| J | F | J & F | link |FPS|
|:-----:|:------:|:----------:|:-:|:-:|:-----:|:----:|:-:|
| STM-Cycle | Resnet18 | DALI | 65.3 | 70.8 | 68.1 | [Google Drive](https://drive.google.com/file/d/1R4RkDPfrNz8JzJNNqFhYIPhbruunNnp5/view?usp=sharing)|14.8|
| STM-Cycle | Resnet50 | PIL | 70.5 | 76.3 | 73.4 | [Google Drive](https://drive.google.com/file/d/1tSTNBeqa9hyKBPX6NzL1N7EgkWAg_2cv/view?usp=sharing)|9.3|

## Running
Appending the root folder to the search path of python interpreter
```bash
export PYTHONPATH=${PYTHONPATH}:./
```

To train the STM network, run following command.
```bash
python3 train.py --cfg config.yaml OPTION_KEY OPTION_VAL
```

To test the STM network, run following command
```bash
python3 test.py --cfg config.yaml initial ${PATH_TO_MODEL} OPTION_KEY OPTION_VAL
```
The test results will be saved as indexed png file at `${ROOT}/${output_dir}/${valset}`.

To run a segmentation demo, run following command
```bash
python3 demo/demo.py --cfg demo/demo.yaml OPTION_KEY OPTION_VAL
```
The segmentation results will be saved at `${output_dir}`.

## Acknowledgements
This codebase borrows the code and structure from [official STM repository](https://github.com/seoungwugoh/STM)

## Reference
The codebase is built based on following works
```Bibtex
@InProceedings{Oh_2019_ICCV,
author = {Oh, Seoung Wug and Lee, Joon-Young and Xu, Ning and Kim, Seon Joo},
title = {Video Object Segmentation Using Space-Time Memory Networks},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}

@InProceedings{Li_2020_NeurIPS,
author = {Li, Yuxi and Xu, Ning and Peng Jinlong and John See and Lin Weiyao},
title = {Delving into the Cyclic Mechanism in Semi-supervised Video Object Segmentation},
booktitle = {Neural Information Processing System (NeurIPS)},
year = {2020}
}
```

