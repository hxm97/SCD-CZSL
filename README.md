# SCD-CZSL
This is the PyTorch code of our AAAI 2023 paper - Leveraging Sub-Class Discrimination for Compositional Zero-shot Learning.
We provide the training and testing code implementation of our method on both UT-Zappos and C-GQA dataset.
<p align="center">
  <img src="img.png" />
</p>

## Setup 

1. Download the images and annotations for UT-Zappos dataset, please run:
```
   bash ./utils/download_data.sh DATA_ROOT
   mkdir logs
```

## Train
2. Using the default parameters to train a model, please run:
```
    python train.py --config CONFIG_FILE
```
For other experiment settings, please change the hyper-parameters in flags.py.

## Test
3. To test the AUC, Harmonic Mean of a model, please run:
```
    python test.py --logpath LOG_DIR
```
where `LOG_DIR` is the directory containing the logs of a model.

## References
If you find this code helpful, please cite
```
@inproceedings{xiaoming2023leveraging,
  title={Leveraging Sub-Class Discrimination for Compositional Zero-shot Learning},
  author={Xiaoming, Hu and Zilei, Wang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2023},
  organization={IEEE}
}
```
## Acknowledgment
We thank the following repos providing helpful components/functions in our work.

- [CGE](https://github.com/ExplainableML/czsl)

- [OADis](https://github.com/nirat1606/OADis))
