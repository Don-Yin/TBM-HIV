# Setup
| Steps | Do                                                                                                         |
|-------|------------------------------------------------------------------------------------------------------------|
| 1     | Make conda env & activate                                                                                  |
| 2     | python==3.10.6; `pip install -r requirements.txt`                                                          |
| 3     | change the `PATH_IMAGE_FOLDER`` and `PATH_LABEL_CSV`` variables in the `train_test_split_images.py` script |
| 4     | run `python train_test_split_images.py` for put the train / valid / test images in place                   |
| 5     | change the python path to that of the env in `tune_single.sh`                                              |
| 6     | run `source jade_sub.sh`                                                                                   |
