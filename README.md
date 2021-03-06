# 🪄 W&B Minimal PyTorch Tutorial

This tutorial is also accompanied with a PyTorch source code, it can be found in `src` folder. Furthermore, all plots and metrics that I mentioned here can be found here in [this link](https://wandb.ai/mrpositron/wandb_tutorial).

You can also run the code with wandb. First you shoule go to a `src` directory, and run the following command:

```
python main.py 
```

## 0. About W&B.

Machine learning *experiment tracking*, *dataset versioning*, and *model evaluation*.


## 1. Setting up.

1. Create an account on [wandb.ai](https://wandb.ai).
2. Install wandb.
```
pip install wandb
```
3. Link your machine with your account.  When logging in you should enter your private API key from [wandb.ai](https://wandb.ai/authorize).
```
wandb login
```

## 2. Start a new run.

```
import wandb
wandb.init(project="my-funny-project")
```

`wandb.init(·)` starts the tracking system metrics and console logs.


## 3. Start to track metrics.

Different metrics like loss, accuracy can be easily done with `wandb.log()` command. For example,

```
wandb.log({'accuracy': train_acc, 'loss': train_loss})
```

By default, `wandb` plots all metrics in one section. If you want to divide sections as for a training, validation, etc. You can just simply add a section name to the metric name by slash.

For example, if you had two losses, training and validation losses. You can split sections as follows:

```
wandb.log({'train/loss': train_loss, 'val/loss': val_loss})
```


## 4. Track hyperparameters.
When using `argparse`, you can use the command below and easily track hyperparameters you have used.
```
wandb.config.update(args) # adds all of the arguments as config variables
```
There are also other ways to save configuration values. For example, you can save configurationsa as a dictionary and pass it. Check more details [here](https://docs.wandb.ai/guides/track/config).


## 5. Track and visualise your weights and gradients.

Add `wandb.watch(model, log = 'all' )` to track gradients and parameters weights.

Visualisation of weights:

![Weights Visualisation](https://github.com/MrPositron/wandb_tutorial/blob/main/images/parameters_example.png)

Visualisation of gradients:

![Gradients Visualisation](https://github.com/MrPositron/wandb_tutorial/blob/main/images/gradients_example.png)

## 6. Tune hyperparameters.

1. Create a sweep configuration file, `sweep.yaml`. 

For example it may look like this:

```
program: train.py
method: bayes
metric:
  name: validation_loss
  goal: minimize
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
  optimizer:
    values: ["adam", "sgd"]
```

2. Initialize a sweep.

Run the following command:
```
wandb sweep sweep.yaml
```

3. Launch agent(s)

```
wandb agent your-sweep-id
```
W&B will present some cool visualisations like this:
![Sweep Example](https://github.com/MrPositron/wandb_tutorial/blob/main/images/sweep_example.png)
