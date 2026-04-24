import traceback
import logging
import json
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    
    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    if args["prefix"] in ["", " "]:
        args["prefix"] = args["model_name"]
    if "project_name" not in args.keys() or args["project_name"] == "":
        args["project_name"] = args["prefix"]
    logs_name = "logs/{}/{}/{}/{}".format(args["dataset"], init_cls, args['increment'],args["project_name"])
    args["logs_name"] = logs_name
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)
    if not os.path.exists(logs_name + "/args"):  
        os.makedirs(logs_name + "/args")        # to store args
        
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        
        _train(args)
        
        
def _train(args):

    logfilename = "{}--{}".format(
        args["prefix"],
        args["seed"]
    )
    args["logfilename"] = logfilename
    
    # save args
    with open(args["logs_name"] + f'/args/{args["logfilename"]}.json', 'w') as json_file:
        json.dump(args, json_file, indent=4)

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)
    
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    logging.debug("Start to get model...")
    model = factory.get_model(args["model_name"], args)
    
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []

    nun_tasks = data_manager.nb_tasks
    if "max_task" in args and args["max_task"] > 0: 
        nun_tasks = min(nun_tasks, args["max_task"])
        
    for task in range(nun_tasks):
        logging.info("All params: {:,}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {:,}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        
        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]    
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            # print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            # print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
            
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN-AA):', cnn_accy["top1"])
            logging.info("Cumulative Accuracy (CNN-CA): {} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

        # if 'print_forget' in args.keys() and args['print_forget'] is True:
        if len(cnn_matrix) > 1:     # start from second task
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(cnn_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            logging.info(f'Accuracy Matrix (CNN): \n{np_acctable}')
            logging.info('Forgetting (CNN-FF): {}'.format(forgetting))
            
        if len(nme_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(nme_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            logging.info(f'Accuracy Matrix (NME): \n{np_acctable}')
            logging.info('Forgetting (NME-FF): {}'.format(forgetting))


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = "cpu"      # torch.device("cpu")
        else:
            device = "cuda:{}".format(device)       # torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))

