# FlowVOS: Weakly-Supervised Visual Warping for Detail-Preserving and Temporally Consistent Single-Shot Video Object Segmentation

This is the official code repository for "FlowVOS: Weakly-Supervised Visual Warping for Detail-Preserving and Temporally Consistent Single-Shot Video Object Segmentation", _BMVC_ 2021. You can also find our project page [here](http://web.stanford.edu/~jxgong/flowvos.html), and the BMVC conference website for this paper [here](https://bmvc2021-virtualconference.com/conference/papers/paper_0363.html).

# Requirements
The detailed setup and installation instructions can be found in the main and dataset README documents in the `stm` subfolder of this repository.

# Dataset
This repository only supports experiments with the DAVIS17 and Youtube-VOS datasets. See further details in the `stm` subfolder.

# Training and testing
## Training
Navigate to the `stm` directory to use the `train.py` training script. An example command for training is below. See further details for training options in the `stm` subfolder.

`python train.py --cfg config.yaml checkpoint ./checkpoints/<your_model_name>/ output_dir output_<your_model_name> initial <path_to_initial_weights> freeze_seg_every 5`

Please reference the training details in our paper. We first train the base model for 240 epochs without the flow module, then use this base model to initialize another training run, where we set `freeze_seg_every` to 5.

## Testing
Navigate to the `stm` directory to use the `test.py` inference script. An example command for testing is below. See further details in the `stm` subfolder.

`python3 test.py --cfg config.yaml initial ./checkpoints/<your_model_pth> output_dir <output_subfolder_name> valset VOS`

# Acknowledgements
This codebase borrows the code and structure from the [official STM-cycle repository](https://github.com/lyxok1/STM-Training).

# Citation
If you use our work, please consider citing our paper:
```Bibtex
@InProceedings{gong2021flowvos,
  author = {Gong, Julia and Holsinger, F. Christopher and Yeung, Serena},
  title = {FlowVOS: Weakly-Supervised Visual Warping for Detail-Preserving and Temporally Consistent Single-Shot Video Object Segmentation},
  booktitle = {Proceedings of the British Conference on Machine Vision (BMVC)},
  month = {November},
  year = {2021}
}
```
