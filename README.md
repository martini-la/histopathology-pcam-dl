# Histopathologic cancer detection: CNNs vs. a Vision Transformer built from scratch

Course project (deep learning, 2024). We compared several architectures for detecting metastatic tissue in small histopathology patches, using the public **PatchCamelyon (PCam)** data as distributed in the Kaggle competition [Histopathologic Cancer Detection](https://www.kaggle.com/competitions/histopathologic-cancer-detection) (lymph-node sections from breast-cancer patients, derived from the Camelyon16 challenge; no duplicate images).

**Authors:** Martín López, Eduardo Rodríguez, José Pinto.

> This is student coursework, not a state-of-the-art result. The limitations below are part of the project. Please read them before using any number.

## What is in this repo

| Path | Content |
|---|---|
| `notebooks/visualtransformer_v3.ipynb` | A Vision Transformer (ViT-Base-like: 12 blocks, 12 heads, 768-d embeddings, 16×16 patches on 96×96 images) written from scratch in PyTorch, with the training loop. This is **Martín López's part**. |
| `docs/project-presentation-es.pdf` | Course presentation (in Spanish): literature review, methods and results for ResNet-18, SqueezeNet 1.0 and the ViT. Some figures come from the cited papers and are credited in the slides. |

The ResNet-18 and SqueezeNet code is **not included** in this repository; their results appear only in the presentation.

## Results (from the presentation)

Test set of 44,005 patches (split of the de-duplicated PCam data: 176,020 train / 44,005 validation / 44,005 test):

| Model | Setup | Test accuracy* |
|---|---|---|
| ResNet-18 (PyTorch) | batch 64, Adam, lr 1e-3, 8 epochs | 95.8 % |
| SqueezeNet 1.0 | batch 32, Adam, lr 1e-4, 16 epochs | 95.0 % |
| ViT from scratch (this notebook) | Adam, lr 3e-4, 15 epochs, 80/20 split of the Kaggle training set | train 0.88 / validation 0.73 (overfits) |

\*Computed from the confusion matrices in the slides. For reference, the DenseNet-169 paper cited in the presentation reports 0.980 accuracy on a related setup.

## Known limitations

- **Possible data leakage in the ViT notebook.** `CustomDataset` multiplies the dataset by four rotations (0°, 90°, 180°, 270°) **before** `random_split`, so rotated copies of the same patch can land in both training and validation. The 0.73 validation accuracy is therefore, if anything, optimistic. A correct protocol splits by image first and augments only the training set.
- The ViT is trained **from scratch**, without pretraining, on limited compute. It is expected to underperform the CNNs here.
- The ROC curves in the presentation were drawn from hard class predictions, not probabilities, so their AUC (0.95) is not a proper AUC.
- The slides state that the classes are balanced, but the training set is about 40.5 % positive / 59.5 % negative.
- No cross-validation or regularization study was run, although the slides announced them.
- No random seeds are fixed, so runs are not exactly reproducible. Only some cells of the notebook have saved outputs.
- One late cell instantiates the model with `in_chans=4`, while the images are loaded as RGB; it was an exploratory experiment and may fail as is.

## How to run the notebook

The dataset is **not** included (Kaggle competition terms). To run it:

1. Download the data from the Kaggle competition page above.
2. Place `train_labels.csv` (columns `id`, `label`), the `train/` folder (`.tif` images) and the `test/` folder next to the notebook.
3. Install: Python 3, `torch`, `torchvision`, `pandas`, `Pillow`, `matplotlib`, `tqdm`. A CUDA GPU is strongly recommended.

## References

See the last slide of the presentation. Main sources: Cukierski, *Histopathologic Cancer Detection* (Kaggle, 2018); Qiu et al., ICCECE 2021; Murcia-Gomez et al., *Applied Sciences* 2022; Yao et al., *Cancers* 2019; Wakili et al., *Computational Intelligence and Neuroscience* 2022.
