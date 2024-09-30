## Background
These datasets are used for the long sequence time-series problem. All datasets have been preprocessed and they were stored as .csv files. The dataset ranges from 2016/07 to 2018/07.

## Source of data sets
ETT-small contains data from 2 power transformers at 2 sites, including load, oil temperature. It consists of two hourly-level datasets (ETTh) and two 15-minute-level datasets (ETTm). Each of them contains load characteristics from July 2016 to July 2018 for seven oil and power transformers.

## Information about the data
We donated two years of data, in which each data point is recorded every minute (marked by m), and they were from two regions of a province of China, named ETT-small-m1 and ETT-small-m2, respectively. Each dataset contains 2 year * 365 days * 24 hours * 4 times = 70,080 data point. Besides, we also provide the hourly-level variants for fast development (marked by h), i.e. ETT-small-h1 and ETT-small-h2. Each data point consists of 8 features, including the date of the point, the predictive value "oil temperature", and 6 different types of external power load features.

We use the .csv file format to save the data. The first line (8 columns) is the horizontal header and includes "date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL" and "OT". The detailed meaning of each column name is shown in the table.

| Field | date | HUFL | HULL | MUFL | MULL | LUFL | LULL | OT |
| :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
| Description | The recorded **date** |**H**igh **U**se**F**ul **L**oad | **H**igh **U**se**L**ess **L**oad | **M**iddle **U**se**F**ul **L**oad | **M**iddle **U**se**L**ess **L**oad | **L**ow **U**se**F**ul **L**oad | **L**ow **U**se**L**ess **L**oad | **O**il **T**emperature (target) |

## Specificities
### ETTh1/ETTh2:
Number of data input features: 7<br>
Sample length: 17420<br>
time granularity: 1h

### ETTm1/ETTm2:
Number of data input features: 7<br>
Sample length: 69680<br>
time granularity: 15m
