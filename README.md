# SequenceSync-GAN (Paper Coming Soon)

This repository provides the official implementation of our SequenceSync-GAN paper titled:<br/>  _**"SequenceSync-GAN: Preserving Temporal Sequential Consistency in Unsupervised Image-to-Image Translation For Cross-domain CHF Detection"**_

![SequenceSync_Github](https://github.com/user-attachments/assets/f6815666-5fbf-49d5-a729-778d25c8c006)

## Paper

[**Coming Soon**]  <!--(https://www.sciencedirect.com/science/article/abs/pii/S0952197623014392)-->

[Firas Al-Hindawi](https://firashindawi.com)<sup>1</sup>, [Md Mahfuzur Rahman Siddiquee](https://github.com/mahfuzmohammad)<sup>2</sup>, Abhidnya Patharkar<sup>2</sup>, JiaJing Huang<sup>2</sup>, Teresa Wu<sup>2</sup>, [Han Hu](https://scholar.google.com/citations?user=5RgSI9EAAAAJ&hl=en)<sup>3</sup><br/>

<sup>1</sup>King Fahd University of Petroleum & Minerals (KFUPM);<sup>2</sup>Arizona State University; <sup>3</sup>University of Arkansas<br/>

## Abstract

Boiling crisis, or critical heat flux (CHF), is a major issue in thermal engineering. It occurs when the heat transfer from a heated surface to a boiling liquid suddenly drops, causing a rapid increase in surface temperature, potentially leading to catastrophic failures like meltdowns in nuclear reactors. Accurate, non-intrusive CHF detection from boiling images is crucial for heat exchanger safety and efficiency. Many image-based machine learning models were developed to detect CHF. However, the generalizability of these models to predict boiling images from different experimental setups (different domains) is often overlooked. Recently, model generalizability was addressed using unsupervised image-to-image (UI2I) translation but struggled with preserving class consistency during translation, resulting in suboptimal cross-domain performance. To address this, we introduce Sequence Synchronized Generative Adversarial Network (SequenceSync-GAN), a novel UI2I translation framework that preserves temporal sequential consistency to improve cross-domain CHF detection. To achieve this, SequenceSync-GAN introduces several novel contributions: 1) a novel data loading approach that incorporates temporal factors, 2) a new sequential annotation that captures temporal aspects, 3) a new discriminator architecture for temporal sequential discrimination, and 4) two loss functions that enforce sequential consistency and enhance cross-domain classification. Evaluated using two boiling image datasets, SequenceSync-GAN outperforms state-of-the-art UI2I models, demonstrating its effectiveness in generalizing CHF detection models. These findings can improve AI-based safety mechanisms in thermal engineering and can be extended to similar applications, such as defect detection in manufacturing.
## Usage

### Clone Repository

```bash
$ git clone https://github.com/Hindawi91/SequenceSync-GAN.git
$ cd SequenceSync-GAN/
```

### A) Replicate Results

#### 1. Download dataset & models:
<ol type="1">
  <li>Download our <a href="https://www.dropbox.com/scl/fi/0iqury0rhq7v81bu2rmpe/data.rar?rlkey=2a35eenysxl0uq20ou0wea5b5&dl=0" > data </a> to replace the current data folder</li>
  <li>Download our <a href="https://www.dropbox.com/scl/fi/k3oi23tmbu9nrfpezcwxm/base_classifier.rar?rlkey=iobe3kdis949j6xi2e0csn1do&dl=0" > Base Classifier </a> and place it inside the "base_classifier_training/" folder</li>
  <li>Download our <a href="https://www.dropbox.com/scl/fi/vyf26trwrx509knfby1pz/models.rar?rlkey=k0qdmrljrek5cpfszvj9osua1&dl=0" > Saved Checkpoint Models </a> and place them inside the "boiling/models/" folder. (best BA model @ 150000, best AUC model @ 90000</li>
</ol>

#### 2. SequenceSync-GAN Val Data Translation:
make sure to select the model u want (150000 or 90000)
```bash
$ bash val.sh
```

#### 3. SequenceSync-GAN Cross-Domain Classification Testing:
```python
$ python test.py
```


### B) Train Using Your Own Dataset

#### 1. Data Preperation

The folder structure should be as follows:

```python
├─data/ # data root
│ ├─train   # directory for training data
│ │ ├─DomainA   # DomainA Train Images
│ │ │ ├─xxx.jpg
│ │ │ ├─ ......
│ │ ├─DomainB   # DomainB Train Images
│ │ │ ├─yyy.jpg
│ │ │ ├─ ......
│ ├─val   # directory for val data
│ │ ├─DomainA   # DomainA val Images
│ │ │ ├─xxx.jpg
│ │ │ ├─ ......
│ │ ├─DomainB   # DomainB val Images
│ │ │ ├─yyy.jpg
│ │ │ ├─ ......
│ ├─test   # directory for test data
│ │ ├─DomainA   # DomainA test Images
│ │ │ ├─xxx.jpg
│ │ │ ├─ ......
│ │ ├─DomainB   # DomainB test Images
│ │ │ ├─yyy.jpg
│ │ │ ├─ ......
```

#### 2. CNN Base Classifier Training:

<ol type="1">
  <li>Assuming one of the domains you have is labeled</li>
  <li>Go to the base_classifier_training folder</li>
  <li>In the “DS_CNN_Training.py” file, change the “dataset” variable to the source DS directory, then run the Python script.</li>
  <li>Once training is done, the best model would be saved as “CNN - Base Model.hdf5”</li>
  <li>In the “test_DS_on_DS.py” file, change the “dataset” variable to the source DS directory, then run the Python script. Then, run the Python script to test the saved model on the source dataset for sanity check.</li>
</ol>

```bash
$ cd base_classifier_training/
$ python DS_CNN_Training.py
```

#### 3. SequenceSync-GAN Training

Start Training:

```bash
$ bash run.sh
```

#### 4. SequenceSync-GAN Val Data Translation

Once Training is done, you need to generate results from each checkpoint model saved

```bash
$ bash val.sh
```

#### 5. SequenceSync-GAN Cross-Domain Classification Testing

Once validation is done, you need to test cross domain classification from each checkpoint model (Assuming you already have a pre-trained classifier on domain A, other wise go to the CNN Base Classifier Training step below): 

```python
$ python test.py
```

## Citation

If you use this code, please cite our paper:

```bibtex
@unpublished{SequencSync,
  title={Sequence-Sync-GAN: Preserving Temporal Sequential Consistency in UI2I Translation For Cross-domain CHF Detection},
  author={Al-Hindawi, Firas and Siddiquee, Md Mahfuzur Rahman and Wu, Teresa and Patharkar, Abhidnya and Hu, Han },
  note = {Manuscript under review},
  year = {2026}
}
```







