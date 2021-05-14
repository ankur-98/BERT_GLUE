from util import prepare_inputs, compute_metrics
from typing import Any, Dict, Union
import torch

def compute_loss(model, inputs, metric=None, metric_1=None):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    if "labels" in inputs:
        labels = inputs.pop("labels")
    
    outputs = model(**inputs)

    # We don't use .loss here since the model may return tuples instead of ModelOutput.
    try:
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    except KeyError:
        loss = outputs.loss

    logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]

    loss = torch.nn.CrossEntropyLoss()(logits, labels)
    
    if metric is not None: 
        metric = compute_metrics(predictions=logits, references=labels, metric=metric)
    if metric_1 is not None: 
        metric_1 = compute_metrics(predictions=logits, references=labels, metric=metric_1)

    return (loss, metric, metric_1)


def training_step(model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]],
                  optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler, 
                  metric: Any=None, metric_1: Any=None) -> torch.Tensor:
    """
    Perform a training step on a batch of inputs.

    Subclass and override to inject custom behavior.

    Args:
        model (:obj:`nn.Module`):
            The model to train.
        inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.
        optimizer (torch.optim.Optimizer): 
            Optimizer instance for the training loop.
        lr_scheduler (torch.optim.lr_scheduler): 
            LR scheduler instance for the training loop.
        metric (Optional): Metric class forward funtion.
        metric_1 (Optional): Second metric class forward funtion.

            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument :obj:`labels`. Check your model's documentation for all accepted arguments.

    Return:
        :obj:`torch.Tensor`: The tensor with training loss on this batch.
    """
    model.train()

    model.zero_grad()
    loss, metric, metric_1 = compute_loss(model, inputs, metric, metric_1)
    
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    return loss.detach(), metric, metric_1


if __name__ == "__main__":
    
    import torch
    from util import *
    from dataloader import get_dataloader
    from train import training_step

    # Config Settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_checkpoint="bert-base-uncased"
    task = "cola"
    batch_size=64
    steps = 2000
    lr = 1e-4

    # Load DataLoader
    print(f"\nLoading data...")
    train_epoch_iterator = get_dataloader(task, model_checkpoint, "train", batch_size=batch_size)
    
    # Load Pre-trained Model
    from transformers import BertForSequenceClassification
    from model import CustomBERTModel
    print(f"\nLoading pre-trained BERT model \"{model_checkpoint}\"")
    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2
    model = CustomBERTModel(model_checkpoint, num_labels=num_labels).to(device)

    # Define optimizer and lr_scheduler
    Optimizer = create_optimizer(model, learning_rate=lr)
    LR_scheduler = create_scheduler(Optimizer)
    Metric, Metric_1 = get_metrics(task)
    tr_loss = torch.tensor(0.0).to(device)
    tr_metric = []
    tr_metric_1 = []
    
    # Training Loop
    from tqdm.auto import tqdm
    print(f"\nTraining begins in batches of {batch_size}..")
    global_steps = 0
    trange = range(len(train_epoch_iterator))
    pbar = tqdm(trange, initial=global_steps, total=steps)
    for e in range((steps//len(train_epoch_iterator))+1):
        iterator = iter(train_epoch_iterator)
        for step in trange:
            global_steps += 1
            pbar.update()
            
            inputs = prepare_inputs(iterator.next(), device)
            step_loss, step_metric, step_metric_1 = training_step(model, inputs, Optimizer, LR_scheduler, Metric, Metric_1)
            tr_loss += step_loss
            tr_metric.append(torch.tensor(list(step_metric.values())[0]))
            if Metric_1 is not None: tr_metric_1.append(torch.tensor(list(step_metric_1.values())[0]))
            
            step_evaluation = {}
            step_evaluation['loss'] = (tr_loss/global_steps).item()
            step_evaluation[f"{Metric.__class__.__name__}"] = torch.stack(tr_metric)[-1:].mean().item()
            if Metric_1 is not None:
                step_evaluation[f"{Metric_1.__class__.__name__}"] = torch.stack(tr_metric_1)[-1:].mean().item()
            pbar.set_postfix(step_evaluation)
            
            if global_steps == steps: 
                break
