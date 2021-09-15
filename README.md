# W&B Tutorial

This is a gentle guide on how to start using an awesome library called **Weights and Biases**. This tutorial is also accompanied with a PyTorch source code, and plots and metrics can be found in this link:

## 0. About W&B

Machine learning experiment tracking, dataset versioning, and model evaluation.


## 1. Setting up

1. Create an account on wandb.ai.
2. Install wandb.
```
pip install wandb
```
3. Link your machine with your account.  When logging in you should enter your private API key from wandb.ai/authorize.
```
wandb login
```

## 2. Start a new run

```
import wandb
wandb.init(project="my-funny-project")
```

`wandb.init(·)` starts the tracking system metrics and console logs.


## 3. Start to track metrics

Different metrics like loss, accuracy can be easily done with `wandb.log()` comamnd. For example,

```
wandb.log({'accuracy': train_acc, 'loss': train_loss})
```

By default, `wandb` plots all metrics in one section. If you want to divide sections as for a training, validation, etc. You can just simply add a section name to the metric name by slash.

For example, if you had two losses, training and validation losses. You can split sections as follows:

```
wandb.log({'train/loss': train_loss, 'val/loss': val_loss})
```


## 4. Track hyperparameters
When using `argparse`, you can use the command below and easily track hyperparameters you have used.
```
wandb.config.update(args) # adds all of the arguments as config variables
```
There are also other ways to save configuration values. For example, you can save configurationsa as a dictionary and pass it. Check more details: https://docs.wandb.ai/guides/track/config.


## 5. Track your weights and gradients

Add `wandb.watch(log = 'all)` to track gradients and parameters weights.

