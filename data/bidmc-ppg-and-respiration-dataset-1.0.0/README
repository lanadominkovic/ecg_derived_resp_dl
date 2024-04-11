BIDMC Dataset
=============

Table of Contents
=================

  * [The Dataset](#the-dataset)
  * [Revisions](#revisions)
  * [Description](#description)

# The Dataset

The BIDMC Dataset was first report in the following publication:

Pimentel, M.A.F. et al. Towards a Robust Estimation of Respiratory Rate from Pulse Oximeters, IEEE Transactions on Biomedical Engineering, 64(8), pp.1914-1923, 2016. [DOI: 10.1109/TBME.2016.2613124](http://doi.org/10.1109/TBME.2016.2613124) 

In this publication it was used to evaluate the performance of different algorithms for estimating respiratory rate from the pulse oximetry, or photoplethysmogram (PPG) signal. The dataset was extracted from the much larger [MIMIC II matched waveform Database](http://physionet.org/physiobank/database/mimic2wdb/matched/), which was acquired from critically-ill patients during hospital care at the Beth Israel Deaconess Medical Centre (Boston, MA, USA). The 53 recordings within the dataset, each of 8-minute duration, each contain:
  * Physiological signals, such as the PPG, impedance respiratory signal, and electrocardiogram (ECG). These are sampled at 125 Hz.
  * Physiological parameters, such as the heart rate (HR), respiratory rate (RR), and blood oxygen saturation level (SpO2). These are sampled at 1 Hz.
  * Fixed parameters, such as age and gender
  * Manual annotations of breaths. Two annotators manually annotated individual breaths in each recording using the impedance respiratory signal.

The dataset is distributed in three formats:
  1) Matlab (r) format, in a manner which is a compatible with the [RRest Toolbox of respiratory rate algorithms](http://peterhcharlton.github.io/RRest);
  2) CSV (comma-separated-value) format;
  3) WFDB (WaveForm DataBase) format, which is the standard format used by [PhysioNet](https://www.physionet.org/).

For more information about the dataset, please contact the authors at:
marco.pimentel@eng.ox.ac.uk, peter.charlton@kcl.ac.uk .

# Revisions

R1: 	2017-Sept-24 		initial release
R2:     2018-Apr-30         second release

## Description
### Matlab (r) Format
The *bidmc_data.mat* file contains the following subset of the dataset in a single Matlab (r) variable named *data*. The following are provided for each of the 53 recordings:
* *ekg*:   Lead II ECG signal. Each signal is provided in a structure, where the *v* field denotes the signal values, and *fs* is the sampling frequency.
* *ppg*:   Photoplethysmogram signal
* *ref.resp_sig.imp*:  Impedance respiratory signal
* *ref.breaths*:  Manual annotations of breaths provided by two independent annotators. A vector of sample numbers is provided, which correspond to the signal sample numbers.
* *ref.params*:  Physiological parameters: *rr* (respiratory rate, derived by the monitor from the impedance signal, breaths per minute), *hr* (heart rate, derived from the ECG, beats per minute), *pr* (pulse rate, derived from the PPG, beats per minute), *spo2* (blood oxygen saturation level, %).
* *fix*: A structure of fixed variables, including: *id* (the MIMIC II matched waveform database subject ID and recording identifier), *loc* (the ward location), and *source* (the URLs from which the original data were downloaded).

### CSV Format

Separate CSV files are provided for each recording (where ## is the subject number), containing:
* *bidmc_##_Breaths.csv*: Manual breath annotations
* *bidmc_##_Signals.csv*: Physiological signals
* *bidmc_##_Numerics.csv*: Physiological parameters
* *bidmc_##_Fix.txt*: Fixed variables

### WFDB Format

Five files are provided for each recording (where ## is the subject number):
* *bidmc##.breath*: Manual breath annotations
* *bidmc##.dat*: Waveform data file
* *bidmc##.hea*: Waveform header file
* *bidmc##n.dat*: Numerics data file
* *bidmc##n.hea*: Numerics header file

Further details on the contents of each file are provided [here](https://physionet.org/tutorials/creating-records.shtml).