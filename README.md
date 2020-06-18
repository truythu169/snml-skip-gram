# Applying information criteria in Skip-gram dimensionality selection

Implementation of Skip-gram Dimensionality Selection via information criteria (SNML, AIC, BIC).

### Requirement
Please make sure your computer has installed these programs below:
```
python3.7
pip
```

### Set up
* Install dependencies
```
pip install -r requirements.txt
```

### Google cloud storage (GCS) set up
We train models on multiple server but save the result on a GCS bucket. Please create an `env.ini` file to store access to the bucket.
The `env.ini` file should be place in the root directory. The file should include content as following:
```
[GCS]
sync = no
project_id = xxx
bucket = xxx
app_credential = xxx
```
Configs:
- sync: set to `no` if you do not want to us GCS
- project_id, bucket: information of the bucket
- app_credential: path to json credential to access GCS

## 1. Preprocess data
### 1.1. Artificial data
Artificial data is generated using jupyter notebooks. 
Please refer to to notebooks below for the details of the data generation process.
- Data for original Skip-Gram model:
```
notebooks/Generate context distributions.jpynb
```
- Data for Skip-Gram Negative Sampling model:
```
notebooks/Generate context distributions - SGNS.jpynb
```


### 1.2. Text data
Run prepocess.py file to prepocess data. This file takes .txt file as input.
Please remove special characters such as .,:? etc in the text file.  
Parameters:
- input: input text file path
- output: output directory
- batch_size: batch size of the process
- window_size: the window size in Skip-Gram model
Example:
```
python preprocess.py --input text8 --output data/text8 --batch_size 1000 --window_size 5
```
Others parameter for preprocessing such as subsampling threshold can be set in config.ini.

## 2. Train Skip-gram
Data after preprocess step can be use to train Skip-gram. Training commands are described as below:

### 2.1. Train original Skip-Gram model
Original Skip-Gram model should be trained using GPUs, we use tensorflow to train this model.
Run tf_based/train.py to train this model.  
Example:
```
python tf_based/train.py --input_path data/text8/ --batch_size 10 --output_path output/text8/ --epochs 1 --n_embedding 5
```
See `config.ini` and `tf_based/train.py` for more parameters settings.

### 2.2. Train Skip-Gram model with Negative Sampling
Skip-Gram Negative Sampling model is trained with numpy. Training process need context distribution to sample negative samples.
Context distribution can be achieved by runing: `utils/context_distribution_from_raw.py`.  
Run np_based/train.py to train this model.  
Example:
```
python np_based/train.py --input_path data/text8/ --batch_size 10 --output_path output/text8/ --epochs 1 --n_embedding 5
```
See `config.ini` and `np_based/train.py` for more parameters settings.

## 3. Estimate AIC & BIC
Estimating AIC & BIC for original Skip-Gram and Skip-Gram Negative Sampling by following programs:

- original Skip-Gram:
```
python tf_based/run_aic_bic.py
```

- Skip-Gram Negative Sampling:
```
python np_based/run_aic_bic.py
```

See each python file for parameters setting.

## 4. Estimate SNML codelength
Estimating SNML for original Skip-Gram and Skip-Gram Negative Sampling by following programs:

- original Skip-Gram:
```
python tf_based/snml/tf_based/train_snml.py
```

- Skip-Gram Negative Sampling:
```
python np_based/train_snml.py
```

Parameters:
- model: output directory of trained model
- context_path: path to context distribution
- snml_train_file: data file, which each element will be estimated codelength
- scope: how many data elements will be estimated
- epochs: the number of epochs in gradient descent while training data in SNML 
- n_context_sample: number of samples in importance sampling
- learning_rate: learning rate in gradient descent while training data in SNML
- continue_from: (do not set this in the first run) the number of element in the last run to continue 
- continue_scope: (do not set this in the first run) continue from previous run, this parameter state number of scope in the previous run 