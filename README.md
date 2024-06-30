# SequenceSync-GAN (Paper Coming Soon)

This repository provides the official implementation of our SequenceSync-GAN paper submitted to the Expert Systems With Applications Journal titled:<br/>  _**"SequenceSync-GAN: Preserving Temporal Sequential Consistency in Unsupervised Image-to-Image Translation For Cross-domain CHF Detection"**_

![sequencesync-gan](https://github.com/Hindawi91/SequenceSync-GAN/assets/38744510/2951b573-7eb5-47b2-a1f7-d14feef85cfa)

## Paper

[**Coming Soon**]  <!--(https://www.sciencedirect.com/science/article/abs/pii/S0952197623014392)-->

[Firas Al-Hindawi](https://firashindawi.com)<sup>1</sup>, [Md Mahfuzur Rahman Siddiquee](https://github.com/mahfuzmohammad)<sup>1</sup>, Abhidnya Patharkar<sup>1</sup>, Teresa Wu<sup>1</sup>, [Han Hu](https://scholar.google.com/citations?user=5RgSI9EAAAAJ&hl=en)<sup>2</sup><br/>

<sup>1</sup>Arizona State University; <sup>2</sup>University of Arkansas<br/>

## Abstract

Boiling crisis, or critical heat flux (CHF), is a major issue in thermal engineering. It occurs when the heat transfer from a heated surface to a boiling liquid suddenly drops, causing a rapid increase in surface temperature potentially leading to severe failures like thermal breakdowns in semiconductors or meltdowns in nuclear reactors. Accurate, non-intrusive detection of CHF from boiling images is essential for the safe operation and effective design of heat exchangers. Most previous machine learning efforts have focused on detecting CHF within the same experimental setup or domain. Detecting out-of-distribution (OOD) CHF is challenging as models must generalize beyond their training conditions to recognize variations in materials, configurations, and environments. Existing computer vision works addressing OOD CHF detection, which utilizes off-the-shelf unsupervised image-to-image translation (UI2I) models, have shown limitations in preserving class consistency during translation due to a lack of class annotations during training, resulting in poor cross-domain classification performance. To address this, we introduce SequenceSync-GAN, a novel UI2I translation framework that encourages accurate class translations across domains by preserving temporal sequential consistency. To achieve this. SequenceSync-GAN introduces several novel contributions including 1) an innovative data loading approach that incorporates temporal factors in a simplified way contrary to previous complicated and computationally expensive methods, 2) a new level of annotation (sequential annotation) that smartly captures the temporal factor, 3) a new discriminator architecture that enables temporal sequential discrimination, and 4) two loss functions that capitalize on the sequential annotation to enforce temporal sequential consistency and overall cross-domain classification performance. SequenceSync-GAN is evaluated using two boiling image datasets against state-of-the-art UI2I translation models, demonstrating its efficacy in addressing the challenges of OOD CHF detection and showcasing its potential for practical application in thermal engineering and other cross-domain applications.

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









