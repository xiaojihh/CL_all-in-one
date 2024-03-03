# Continual All-in-One Adverse Weather Removal with Knowledge Replay on a Unified Network Structure [TMM 2024]



> [**Continual All-in-One Adverse Weather Removal with Knowledge Replay on a Unified Network Structure**]<br>
> [De Cheng*], [Yanling Ji*], [Dong Gong], [Nannan Wang], [Junwei Han], [Dingwen Zhang]



## Requirements
- Python 3.6+  
```pip install -r requirements.txt```

## Experimental Setup
Our code requires three datasets: OTS, Rain100H, Snow100K
### Dataset
We recommend putting all datasets under the same folder (say $datasets) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure should look like:

```
$datasets/
|–– RESIDE/
    |–– OTS_beta/
        |–– hazy/
        |–– clear/
    |–– SOTS/
        |–– outdoor/
            |–– hazy/
            |–– clear/
|–– Rain100H/
    |–– train
        |–– rain
        |–– norain
    |–– test
        |–– rain
        |–– norain
|–– Snow100K
    |–– train
        |–– synthetic
        |–– gt
    |–– test
        |–– Snow100K-M
            |–– synthetic
            |–– gt
```
First, run `python patch.py` to patch Rain100H.


## Usage
If you want to reproduce the results mentioned in our paper, run
```
python main.py --task_order haze rain snow --memory_size 500 --exp_name haze_rain_snow --eval_step 20000 --device cuda:0

```
We provide our training checkpoints and you can continue training using the `--resume` hyperparameter.


### Hyperparameters

The meaning of hyperparameters in the command line is as follows:

| params              | name                                            |
| -----------------   | ----------------------------------------------- |
| --task_order        | task order for dehazing, deraining, desnowing   |
| --memory_size       | memory size                                     |
| --exp_name          | experiment name                                 |
| --eval_step         | each step for eval                              |
| --resume            | training from previous tasks                    |

If you encounter any issues or have any questions, please let us know. 
