# Term Project
This data science project predicts an Acute Kidney Injury (AKI) in patients admitted to hospitals. As training data, we are using the [MIMIC-III Critical Care Database](https://mimic.physionet.org/about/mimic/) which has over 40,000 patients and 58k hospital admissions between 2001 and 2012. The objective is to find a parsimonious model which can help predict the likelihood of AKI in a setting where rapid delivery of information can mean rapid treatment. We will reduce features to the few elements of information that are sufficiently useful to predict likelihood of AKI with a decent degree of accuracy.

### File Structure
The code and the data are located in the same directory. The [demo data](https://physionet.org/content/mimiciii-demo/1.4/) is in a demo-data/ folder on my local and the [full data files](https://physionet.org/content/mimiciii/1.4/) are located there as well. The following ipython notebooks do a bunch of different things, splitting some data which was too big for memory and feature engineering/exploration. Each jupyter notebook has its own section below.

term-project/
├── data/   (not committed)
│   ├── data-file-1...
│   └── data-file-2...
├── demo-data/   (not committed)
│   ├── demo-data-file-1...
│   └── demo-data-file-n...
├── split-data/   (not committed)
│   └── chart-events/
│       └── each-bin-of-chart-data...
├── dev_feature_engineering.ipynb (1)
├── feature_engineering_utilities.py (2)
├── split_charts_data.ipynb (3)
├── prod_feature_engineering.ipynb (4)
└── readme.md

### Feature Engineering (dev_feature_engineering.ipynb)
In order to do the feature engineering



### Splitting Data (split_charts_data.ipynb)
The `chartevents` table was too big for memory. I could have used the AWS Environment (with Athena and the Sagemaker notebooks but wouldn't have been able to preserve my work easily here). I also could have used bigquery and the python pandas API for it, but it felt unnecessary. Instead, I took the largest table I needed to parse and split the data by 

# unzipping data
- run the commands in the unzip_move_data text file in the terminal
- data in the data/ folder are the csv's of the sample, when we move to postgres, we can switch out the datasets but we can use this to dev