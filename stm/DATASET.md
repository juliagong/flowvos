# NOTE: The below README is the original dataset README from the [STM-cycle](https://github.com/lyxok1/STM-Training) code base, with the exception of a note regarding model outputs and a few minor spelling changes.

## Dataset

To run the training and testing code, we require the following data organization format
```
${ROOT}--
        |--${DATASET1}
        |--${DATASET2}
        ...
```
The `ROOT` folder can be set in `libs/dataset/data.py` by setting `ROOT = "path/to/root"`. Each sub-directory ${DATASET} should be the name of one specific dataset (e.g. DAVIS17 or Youtube-VOS) and contain all video and annotation data.

The model outputs will also be saved as subfolders to this folder (see main STM-cycle README).

### DAVIS Organization

To run the training script on davis16/17 dataset, please ensure the data is organized as following format
```
DAVIS16(DAVIS17)
      |----JPEGImages
      |----Annotations
      |----data
      |------|-----db_info.yaml
```
Where `JPEGImages` and `Annotations` contain the 480p frames and annotation masks of each video. The `db_info.yaml` contains the meta information of each video sequences and can be found at the davis evaluation [repository](https://github.com/fperazzi/davis-2017/blob/master/data/db_info.yaml).

### Youtube-VOS Organization
To run the training script on youtube-vos dataset, please ensure the data is organized as following format
```
Youtube-VOS
      |----train
      |     |-----JPEGImages
      |     |-----Annotations
      |     |-----meta.json
      |----valid
      |     |-----JPEGImages
      |     |-----Annotations
      |     |-----meta.json 
```
Where `JPEGImages` and `Annotations` contain the frames and annotation masks of each video.

### Custom Dataset

To run the codebase on custom dataset, please see the [template](./libs/dataset/template.py) and implement related interface and alias.

### DALI Backend Support

Nvidia-dali backend loader is implemented for training on davis and youtube-vos. If you want to use the dali loader for efficient frame decode and transformation, you can install [DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html) first and set `data_backend: 'DALI'` in `config.yaml` or `data_backend DALI` in command line. 

**Note: DALI backend is only for training now**
