# Master Thesis



## Uncertainty Estimation in the Segmentation of Brain Magnetic Resonance Angiograms

This thesis focuses on estimating epistemic uncertainty in the segmentation of cerebrovascular structure from brain MRAs using an efficient ensemble model by [Lee et.al.](https://arxiv.org/abs/2005.10754). Epistemic uncertainty is the uncertainty associated with model's parameters. By estimating uncertainty, we can make deep-learning models more explainable and trust-worthy, thus encouraging their real-world applications. 

## Dataset
Download COSTA dataset from [here](https://zenodo.org/records/11025761) (need to request for permission)
- Organize the directory : 
```
data_path
├── train_input
├── train_label
├── test_input
├── test_label
├── val_input
└── val_label
```
OR

Download publicly Available TubeTk dataset from [here](https://rwth-aachen.sciebo.de/s/svPc2wrC3oh1kTU) 

## Clone repository and install requirements
Clone the repository:
```bash
git clone https://git.rwth-aachen.de/omini1997/master-thesis.git
```

Navigate to the directory and install requirements
```bash
pip install -r requirements.txt
```

## Run baseline
Run baseline using train_baseline.sh. To directly run inference using trained baseline in trained-model, comment training code in baseline.py
```bash
sbatch train_baseline.sh
```

## Run ensemble
Run ensemble using train_ensemble.sh. To run train step 2 directly, use ensemble step-1 model in trained-models and comment out step 1 training in ensemble.py.  To directly run inference using ensemble final model in trained-model, comment training code in ensemble.py.
```bash
sbatch train_ensemble.sh
```

## Evaluation
Calculate Dice score, clDice, sensitivity and precision using metrics.py. Pass output label path as out_path, test label path as mask_path.


