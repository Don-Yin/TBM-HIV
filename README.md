# Setup
| Steps | Do                                                                                                         |
|-------|------------------------------------------------------------------------------------------------------------|
| 1     | Make conda env & activate                                                                                  |
| 2     | python==3.10.6; `pip install -r requirements.txt`                                                          |
| 3     | change the `PATH_IMAGE_FOLDER` and `PATH_LABEL_CSV` variables in the `train_test_split_images.py` script |
| 4     | run `python train_test_split_images.py` for put the train / valid / test images in place                   |
| 5     | change the python path to that of the env in `tune_single.sh`                                              |
| 6     | run `source jade_sub.sh`                                                                                   |

# Note
This codebase is part of an individual research project titled *Phenotyping Brain Changes in HIV and non-HIV Patients with Tuberculosis Meningitis.* The starting process is detailed in the setup section above. Please be aware that the project will not run as expected, as the data is not publicly available. The data for this research is derived from two prospective longitudinal studies conducted at Oxford University Clinical Research Unit (OUCRU). For more information, see the following two studies:
- [Study 1](https://doi.org/10.12688/wellcomeopenres.14007.1)
- [Study 2](https://doi.org/10.12688/wellcomeopenres.14006.2)
The code is made available to promote transparency and accessibility in research methodology.

## Contact
For inquiries, please contact the author at [author's email](mailto:don_yin@kcl.ac.uk).