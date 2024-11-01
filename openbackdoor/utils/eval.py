from openbackdoor.victims import Victim
from .log import logger
from .metrics import classification_metrics, detection_metrics
from typing import *
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
from openbackdoor.utils.visualize import plot_attention


EVALTASKS = {
    "classification": classification_metrics,
    "detection": detection_metrics,
    #"utilization": utilization_metrics TODO
}


def evaluate_classification(model: Victim, eval_dataloader, metrics: Optional[List[str]]=["accuracy"], plot=False, info=''):
    # effectiveness
    results = {}
    dev_scores = []
    main_metric = metrics[0]
    for key, dataloader in eval_dataloader.items():
        results[key] = {}
        logger.info("***** Running evaluation on {} *****".format(key))
        eval_loss = 0.0
        nb_eval_steps = 0
        model.eval()
        outputs, labels = [], []
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_inputs, batch_labels = model.process(batch)
            with torch.no_grad():
                batch_outputs = model(batch_inputs, output_attentions=plot)
            if plot and batch['poison_label'][0]==1 and len(batch_inputs['input_ids'][0])<15:
                tokens = model.tokenizer.convert_ids_to_tokens(batch_inputs['input_ids'][0])
                plot_attention(batch_outputs['attentions'], tokens, info)
                plot = False
            outputs.extend(torch.argmax(batch_outputs.logits, dim=-1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
        logger.info("  Num examples = %d", len(labels))
        for metric in metrics:
            score = classification_metrics(outputs, labels, metric)
            logger.info("  {} on {}: {}".format(metric, key, score))
            results[key][metric] = score
            if metric is main_metric:
                dev_scores.append(score)

    return results, np.mean(dev_scores)


def evaluate_step(model: Victim, dataloader, metric: str):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            batch_inputs, batch_labels = model.process(batch)
            output = model(batch_inputs).logits
            preds.extend(torch.argmax(output, dim=-1).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
    score = classification_metrics(preds, labels, metric=metric)
    return score

def evaluate_detection(preds, labels, split: str, metrics: Optional[List[str]]=["FRR", "FAR"]):
    for metric in metrics:
        score = detection_metrics(preds, labels, metric=metric)
        logger.info("{} on {}: {}".format(metric, split, score))
    return score    
