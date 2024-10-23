# Defend
import os
import json
import argparse
import time

import openbackdoor as ob
from openbackdoor.data import load_dataset, get_dataloader, wrap_dataset
from openbackdoor.victims import load_victim
from openbackdoor.attackers import load_attacker
from openbackdoor.defenders import load_defender
from openbackdoor.utils import set_config, logger, set_seed, set_config_detail
from openbackdoor.utils.visualize import display_results
import numpy as np
import os
import datetime

# import pyarrow.lib

# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config_path", type=str, default="./configs/loss_config.json")
#     parser.add_argument("--seed", type=int, default=42)
#     args = parser.parse_args()
#     return args


def main(config=None, config_victim=None, config_attacker=None, config_defender=None, config_dataset=None, path=None):
    # choose a victim classification model
    if config is not None:
        victim = load_victim(config["victim"])
        # choose attacker and initialize it with default parameters
        attacker = load_attacker(config["attacker"])
        defender = load_defender(config["defender"])
        target_dataset = load_dataset(**config["target_dataset"])
        poison_dataset = load_dataset(**config["poison_dataset"])
    elif config_victim is not None and config_attacker is not None and config_defender is not None:
        victim = load_victim(config_victim)
        attacker = load_attacker(config_attacker)
        defender = load_defender(config_defender)
        target_dataset = load_dataset(**config_dataset["target_dataset"])
        poison_dataset = load_dataset(**config_dataset["poison_dataset"])
    else:
        print('no config')
        return

    # target_dataset = attacker.poison(victim, target_dataset)
    # launch attacks
    defender.path = path
    backdoored_model = attacker.attack(victim, poison_dataset, config, defender)

    if config_attacker['train']['visualize']:
        if config_dataset['target_dataset']['name'] == 'sst-2':
            name = 'sst'
        elif config_dataset['target_dataset']['name'] == 'hate-speech':
            name = 'hs'
        else:
            name = 'agnews'
        import pandas as pd
        if mlm:
            df = pd.DataFrame(attacker.mlm_trainer.info)
        else:
            df = pd.DataFrame(attacker.poison_trainer.info)
        # path = os.path.join('./info/agnewsb-labelmix-0.3', '%s-%s-%s-%.1f-%s-%d.csv' %
        #                     (name, attacker.poison_trainer.poison_setting,
        #                      attacker.poison_trainer.poison_method, attacker.poison_trainer.poison_rate,
        #                      attacker.poison_trainer.lr, config_attacker['poisoner']['target_label']))
        path = os.path.join('./info/%s'% name, '%s-%s-%.1f-%s-%d-let-0.2-none-none.csv' %
                            (name, attacker.poison_trainer.poison_method, attacker.poison_trainer.poison_rate,
                             attacker.poison_trainer.lr, config_attacker['poisoner']['target_label']))
        df.to_csv(path)

    if mlm:
        return []
    results = attacker.eval(backdoored_model, target_dataset, defender)
    
    # Fine-tune on clean dataset
    '''
    print("Fine-tune model on {}".format(config["target_dataset"]["name"]))
    CleanTrainer = ob.BaseTrainer(config["train"])
    backdoored_model = CleanTrainer.train(backdoored_model, wrap_dataset(target_dataset, config["train"]["batch_size"]))
    '''
    return results, defender.info


# def run(config_path="./configs/loss_config.json"):
def run(config_path=None, victim=None, attacker=None, defender=None, dataset=None, rate=None, runs=5, flag=''):
    seed = 42
    config = None
    config_victim = None
    config_attacker = None
    config_defender = None
    config_dataset = None
    if config_path is not None:
        with open(config_path, "r") as f:
            config = json.load(f)
        config = set_config(config)
        poison_rate = config['attacker']['poisoner']['poison_rate']
        attacker = config['attacker']['poisoner']['name']
        defender = config['defender']['name']
        dataset = config['poison_dataset']['name']
    else:
        with open("./configs_detail/attackers/%s.json" % attacker, "r") as f:
            config_attacker = json.load(f)
            if rate is not None:
                config_attacker['poisoner']['poison_rate'] = rate
        with open("./configs_detail/defenders/%s.json" % defender, "r") as f:
            config_defender = json.load(f)
        with open("./configs_detail/victims/%s.json" % victim, "r") as f:
            config_victim = json.load(f)
        with open("./configs_detail/datasets/%s.json" % dataset, "r") as f:
            config_dataset = json.load(f)
        config_victim, config_attacker, config_defender, config_dataset = (
            set_config_detail(config_victim, config_attacker, config_defender, config_dataset))
        poison_rate = config_attacker['poisoner']['poison_rate']

    entries = []
    t_lst = []
    for r in range(runs):
        start = time.time()
        set_seed(seed + r)
        results, info = main(config, config_victim, config_attacker, config_defender, config_dataset, path)

        if mlm:
            return
        CACC = results['test-clean']['accuracy']
        ASR = results['test-poison']['accuracy']
        t = time.time() - start
        entries.append([CACC, ASR])
        t_lst.append(t)

        txt = f'{dataset},{poison_rate},{attacker},{defender}'
        txt += f',{CACC:.4f},{ASR:.4f}'
        txt += f',{t:.1f},{r + 1}/{runs}'
        print(txt)

        if silence != 1:
            f = open(detail_file, 'a')
            print(txt, file=f)
            print(info, file=f)
            f.close()

    avg_entry = np.average(np.array(entries), axis=0)
    std_entry = np.std(np.array(entries), axis=0)
    avg_t = np.average(t_lst)

    txt = f'{dataset},{poison_rate},{attacker},{defender}-{flag}'
    txt += f',{avg_entry[0]:.4f},{std_entry[0]:.4f},{avg_entry[1]:.4f},{std_entry[1]:.4f}'
    txt += f',{avg_t:.1f}'
    print(txt)

    if silence != 1:
        f = open(sum_file, 'a')
        print(txt, file=f)
        f.close()


if __name__=='__main__':
    mlm = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--silence', default=1, type=int)
    args = parser.parse_args()

    silence = args.silence      # 1 不把结果写道result里
    sum_file = './results/sum.csv'
    detail_file = './results/detail.csv'

    if silence != 1:
        print('########### Not Scilence #############')
        with open(sum_file, 'a') as f:
            print(f'{datetime.datetime.now()}', file=f)
            print(f'data,poison_rate,attacker,defender,CACC,std,ASR,std,time', file=f)
            f.close()
        with open(detail_file, 'a') as f:
            print(f'{datetime.datetime.now()}', file=f)
            print(f'data,poison_rate,attacker,defender,CACC,std,ASR,std,time,iter', file=f)
            f.close()

    victims = ['plm']
    attackers = ['badnets', 'addsent', 'style', 'syntactic']
    # attackers = ['badnets']
    defenders = ['none', 'lossin', 'onion', 'rap', 'zdefence', 'muscle', 'badacts']
    defenders = ['lossin']
    datasets = ['sst-2', 'hate-speech', 'agnews']
    # datasets = ['agnews']
    poison_rates = [0.1, 0.2, 0.3, 0.4]
    jsons = ["./configs/loss_config.json", "./configs/onion_config.json"]
    # for j in jsons:
    #     run(config_path=j)
    for dataset in datasets:
        for attacker in attackers:
            for defender in defenders:
                victim = victims[0]
                print("RUNNING %s %s %s %s %f" % (victim, attacker, defender, dataset, 0.2))
                path = '%s-%s' % (dataset, attacker)
                run(victim=victim, attacker=attacker, defender=defender, flag='', dataset=dataset, rate=0.2, runs=1)
    # run(victim='plm', attacker='syntactic', defender='none', flag='', dataset='sst-2', rate=0.2, runs=1)