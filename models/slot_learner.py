import copy
import os
import logging
import json
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
# from tqdm import tqdm
from torch.utils.data import DataLoader
from models.base import BaseLearner
from trainer import _set_device
from utils.inc_net import SlotAttentionVitNet
from utils.toolkit import accuracy, tensor2numpy


num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        logging.debug(f"Using SlotAttentionVitNet")
        self._network = SlotAttentionVitNet(args, True)
        
        self.slots = []

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.lr_task_wise_decay = args["lr_task_wise_decay"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        
        if not os.path.exists(os.path.join(args['logs_name'], 'checkpoints')):
            os.makedirs(os.path.join(args['logs_name'], 'checkpoints'))
            
        self.debug_verbose = True 
        self.reset_debug_verbose()
    
    def reset_debug_verbose(self):
        self.debug_verbose = 1
        logging.debug("Debug verbose reset to {}".format(self.debug_verbose))
    
    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self.reset_debug_verbose()
        self._cur_task += 1

        if self._cur_task > 0:
            try:
                if self._network.module.slot is not None:
                    self._network.module.slot.process_task_count()
            except:
                if self._network.slot is not None:
                    self._network.slot.process_task_count()

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # self._network.update_fc(self._total_classes)
        logging.info("Slot learning on {}-{}".format(self._known_classes, self._total_classes))

        self.data_manager = data_manager
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="val", mode="test" )
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

        # try to load the trained model
        need_train = True
        if 'resume' in self.args.keys() and self.args['resume']:
            try:
                checkpoint_name = os.path.join(self.args['logs_name'], 'checkpoints', f"{self.args['logfilename']}_{self._cur_task}.pkl")
                self.load_checkpoint(checkpoint_name)
                need_train = False
            except:
                logging.info("Failed to load model from {}. Training Needed.".format(checkpoint_name))
                need_train = True
        if need_train:
            train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="train")
            self.train_dataset = train_dataset
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
            if len(self._multiple_gpus) > 1:
                logging.info("Multiple GPUs")
                self._network = nn.DataParallel(self._network, self._multiple_gpus)
            self._train(self.train_loader, self.test_loader)
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module
            
            # save slot module
            self.save_checkpoint(os.path.join(self.args['logs_name'], 'checkpoints', 
                                            self.args['logfilename']))
            
        # store slot for eval
        self.slots.append(copy.deepcopy(self._network.slot.state_dict()))

    def load_checkpoint(self, checkpoint_name): 
        logging.info("Loaded model from {}.".format(checkpoint_name))
        checkpoint = torch.load(checkpoint_name, map_location=self._device)
        self._network.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logging.info("With param: \n{}.".format(list(checkpoint['model_state_dict'].keys())))
    
    def save_checkpoint(self, filename):
        original_device = next(self._network.parameters()).device
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": {k: p for k, p in self._network.state_dict().items() if k.startswith('slot')},
        }
        file_name = "{}_{}.pkl".format(filename, self._cur_task)
        torch.save(save_dict, file_name)
        self._network.to(original_device)
        logging.info(f"Save model to {file_name}.")

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        self.data_weighting()
        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def data_weighting(self):       # all 1
        self.dw_k = torch.tensor(np.ones(self._total_classes + 1, dtype=np.float32))
        self.dw_k = self.dw_k.to(self._device)

    def get_optimizer(self):
        if self._cur_task > 0:
            lr = self.init_lr * self.lr_task_wise_decay     # other task
        else:
            lr = self.init_lr           # first task
        if len(self._multiple_gpus) > 1:
            params = list(self._network.module.slot.parameters())
        else:
            params = list(self._network.slot.parameters())
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(params, momentum=0.9, lr=lr,weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(params, lr=lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(params, lr=lr, weight_decay=self.weight_decay)

        logging.info('******************* init optimizer **********************')
        # {num:,} => 1,000,000 with ","
        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in params if p.requires_grad)
        logging.info(f'{total_trainable_params:,} training parameters, len {len(params)}.')

        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            # scheduler = CosineSchedule(optimizer, K=self.args["tuned_epoch"])
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args['tuned_epoch'], eta_min=self.min_lr)
            
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        # prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(range(self.args['tuned_epoch'])):
            self._network.train()
            
            collects = {}
            for i, (_, inputs, targets) in enumerate(train_loader):
                collect = {}
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                # logits
                _, res = self._network(inputs, pen=True)
                
                loss = 0
                # recon loss
                recon_loss = res['recon_loss'].mean()
                collect['recon_loss'] = recon_loss.item()
                loss += recon_loss

                # primitive loss
                if self.args['use_p_reg']:
                    p_loss = self._primitive_loss(res, targets)
                    collect['primitive_losses'] = p_loss.item()
                    loss = loss + self.args['p_reg_coeff'] * p_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                collect['losses'] = loss.item()
                
                # add collect from each batch to collects
                for key, value in collect.items():
                    if isinstance(value, (int, float)):
                        # remove non scalar
                        if key not in collects.keys():
                            collects[key] = 0
                        collects[key] += value

                if self.debug_verbose > 0: 
                    self.debug_verbose -= 1
            
            if scheduler:
                scheduler.step()
            
            # Collect lr
            lr_dict = {}
            for group_id, param_group in enumerate(optimizer.param_groups):
                # logging.info(f"LR: {param_group['lr']}")
                lr_dict[f'train/LR{group_id}'] = param_group['lr']

            # Make name brief
            brief_collects = {}     # e.g., PrimLoss
            for key, v in collects.items():
                if key != 'losses':
                    visual_key = ''     
                    for word in key.split('_'):
                        visual_key = visual_key + word[0].upper()+word[1:4].lower()
                    if visual_key not in brief_collects.keys():
                        brief_collects[visual_key] = 0
                    brief_collects[visual_key] += v

            info = "Task {}, Epoch {}/{} => Loss {:.3f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                collects['losses'] / len(train_loader)
            )
            # Add other metrics in collects
            for name, value in brief_collects.items():
                info = info + ' | {name} {value:.3f}'.format(name=name, value=value / len(train_loader))
            
            # prog_bar.set_description(info)
            logging.info(info)  

        logging.info(info)

    def _wslot_sim(self, weighted_slots): 
        """not normed"""
        sim_mode = self.args['p_reg_sim_mode']      # cos
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        if 'dot' == sim_mode:
            raw_sim = weighted_slots @ weighted_slots.t()
            sim = raw_sim * (
                        self.args['p_reg_temp'] * (weighted_slots.shape[-1] ** -0.5))
        else:
            raw_sim = cos(weighted_slots.unsqueeze(0), weighted_slots.unsqueeze(1))
            sim = raw_sim * (
                self.args['p_reg_temp'])     # [b,b]
        
        return sim
    
    def _label_sim(self, targets):
        """normed label cosine sim"""
        cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        targets_1hot = F.one_hot(targets).float()
        label_sim = cos(targets_1hot.unsqueeze(1), targets_1hot.unsqueeze(0))      # [bs, bs]
        label_sim = label_sim / label_sim.sum(dim=-1, keepdim=True)    # l1-norm
        
        return label_sim
    
    def _primitive_loss(self, res, targets):
        w_slots = res['w_slots']        # [bs, d128]
        weighted_slots = w_slots
        bs = w_slots.shape[0]
        if targets.dim() == 0:
            targets = targets.repeat(bs)
            
        label_sim = self._label_sim(targets)
        sim = self._wslot_sim(weighted_slots)
        p_loss = cross_entropy_with_soft_labels(sim, label_sim)

        return p_loss

    def eval_task(self):
        # no classification task evaluation should be performed on slot learner.
        self._network.to(self._device)
        self._network.eval()
        
        cnn_accy = 0
        nme_accy = None
        
        return cnn_accy, nme_accy

    

class LearnerWrapper:
    """Overwrite _get_loss method of the wrapped learner to use additional loss. 
    """
    def __init__(self, learner_instance):
        self.learner = learner_instance
        self._get_loss_ori = self.learner._get_loss
        self.learner._get_loss = self._get_loss     
        # overwrite learner._get_loss, thus, it will be cal-ed in learner's _init_train function
        
        self.args = self.learner.args
        
        if self.args['use_s_l_reg']:
            self.slot_args = self.load_slot_args()
            self.slot_learner = Learner(self.slot_args)
            self.slot_learner._network.to(self._device)
            
            self.slot_dim = self.slot_args['slot_dim']
        
        self.debug_verbose = True 
        self.reset_debug_verbose()
    
    def __getattr__(self, name): 
        """
        Delegate attribute access to the wrapped Learner instance.

        Args:
            name (str): The attribute name.

        Returns:
            The attribute from the wrapped Learner instance.
        """
        return getattr(self.learner, name)
    
    def reset_debug_verbose(self):
        self.debug_verbose = 1
        logging.debug("Debug verbose reset to {}".format(self.debug_verbose))   
    
    def load_slot_args(self): 
        project_path = self.args["logs_name"]
        # slot model prelearned in project: slots
        slot_project_path = os.path.relpath(project_path + f"/../{self.args['slot_project']}")
        slot_model_name = f"{self.args['slot_prefix']}--{self.args['seed']}"
        
        full_path = slot_project_path + f'/args/{slot_model_name}.json'
        try:
            with open(full_path, 'r') as json_file:
                args = json.load(json_file)
            logging.info(f"Load slot args from {full_path}.")
        except FileNotFoundError:
            logging.info(f"Slot args file not found: {full_path}")
            logging.info(f"Try to load slot args from {self.args['slot_args']}...")
            slot_args = self.args['slot_args']
            with open(slot_args, 'r') as json_file:
                args = json.load(json_file)
            
            init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
            logs_name = "logs/{}/{}/{}/{}".format(args["dataset"], init_cls, args['increment'],args["project_name"])
            args["logs_name"] = logs_name
            args["logfilename"] = slot_model_name
            args["seed"] = self.args["seed"]        # update seed to current seed
            
            if not os.path.exists(logs_name):
                os.makedirs(logs_name)
            if not os.path.exists(logs_name + "/args"):  
                os.makedirs(logs_name + "/args")        # to store args
                
            # save args
            with open(args["logs_name"] + f'/args/{args["logfilename"]}.json', 'w') as json_file:
                json.dump(args, json_file, indent=4)

            logging.info(f"Load slot args from {self.args['slot_args']}.")
        
        logging.info(f"Slot logs_name: {args['logs_name']}")
        logging.info(f"Slot logfilename: {args['logfilename']}")
        
        _set_device(args)     # slot_args['device'] from 0 to cuda
            
        return args
    
    def load_slot_checkpoint(self, task_id): 
        logging.info("Loaded slot model.")
        checkpoint_name = os.path.join(self.slot_args['logs_name'], 'checkpoints', f"{self.slot_args['logfilename']}_{task_id}.pkl")
        # self.slot_learner.to(self._device)
        self.slot_learner.load_checkpoint(checkpoint_name)

    def _ensure_slot_network_device(self, target_device):
        """Keep slot learner network on the same device as training inputs."""
        slot_param = next(self.slot_learner._network.parameters(), None)
        if slot_param is None:
            return
        if slot_param.device != target_device:
            logging.debug(
                "Move slot learner network from %s to %s.",
                slot_param.device,
                target_device,
            )
            self.slot_learner._network.to(target_device)
    
    def _get_loss(self, inputs, targets, *args, **kwargs):         
        """
        Override the _get_loss method of the wrapped Learner.
        This will be called in learner's method.

        Args:
            inputs: Input data.
            targets: Target labels.

        Returns:
            loss, res: The computed loss and additional results.
        """
        loss, res = self._get_loss_ori(inputs, targets, *args, **kwargs)
        
        # obtain additional reg loss
        logits = res['logits']      # should be the logits used to calculate the loss
        
        if self.args['use_s_l_reg']:
            self._ensure_slot_network_device(inputs.device)
            _, slot_collect = self.slot_learner._network(inputs, pen=True)
            # slot_collect contains: 'slots', 'attns', ...
            
            wslots = slot_collect['w_slots']
            collect = {}
            sl_loss = self._slot_logit_reg(wslots, logits, collect)
            # collect contains: 'slot_sim', 'logit_sim'
            
            if 'aux_logits' in res.keys():      # in MEMO, DER
                logits = res['aux_logits']
                sl_loss = sl_loss + self._slot_logit_reg(wslots, logits, collect)
                
            if 'fe_logits' in res.keys():       # in FOSTER
                logits = res['fe_logits']
                sl_loss = sl_loss + self._slot_logit_reg(wslots, logits, collect)
                
            loss = loss + self.args['s_l_reg_coeff'] * sl_loss
            res['slot_logit_reg_loss'] = sl_loss
            
            if self.debug_verbose > 0: 
                logging.debug(f"slot_logit_reg_loss: {sl_loss}")

        if self.debug_verbose > 0: 
            logging.debug(f"loss: {loss}")
            self.debug_verbose -= 1
        return loss, res
    
    def _logit_sim(self, logits): 
        temp = self.args['l_reg_temp']
        logit_sim = torch.matmul(logits, logits.t()) * (
                temp * (logits.shape[-1] ** -0.5))
        
        return logit_sim
        
    def _slot_logit_reg(self, wslots, logits, collect):
        weighted_slots = wslots
        
        # slot sim
        with torch.no_grad():
            slot_sim = self.slot_learner._wslot_sim(weighted_slots)   # before normailzed
        
        ## wslot as 'label', thus, use a min-max norm to sum 1
        mode = self.args.get('s_l_reg_sup_mode', 'minmax')       # 'minmax' 'softmax'
        # # softmax
        if mode == 'softmax':
            normed_slot_sim = F.softmax(slot_sim, dim=-1)
        elif mode == 'minmax':
            # minmax over row to make them positive
            normed_slot_sim = (slot_sim - slot_sim.min(dim=-1, keepdim=True)[0]) / (
                    slot_sim.max(dim=-1, keepdim=True)[0] - slot_sim.min(dim=-1, keepdim=True)[0] + 1e-10)      # max 1, min 0
            normed_slot_sim = normed_slot_sim / normed_slot_sim.sum(dim=-1, keepdim=True)  # l1-norm to sum 1
        else: 
            raise ValueError(f"Unknown s_l_reg_sup_mode: {mode}")
        
        # logit sim
        logit_sim = self._logit_sim(logits)
        
        mode = self.args['s_l_reg_loss_mode']       # kl
        if mode == 'l2':
            loss = F.mse_loss(logit_sim, slot_sim)
        elif mode == 'l1':
            loss = F.l1_loss(logit_sim, slot_sim)
        else:       # kl
            loss = cross_entropy_with_soft_labels(logit_sim, normed_slot_sim)

        collect['slot_sim'] = slot_sim
        collect['logit_sim'] = logit_sim
        
        return loss
    
    def incremental_train(self, data_manager):
        # load slot model before call learner.incremental_train method
        self.reset_debug_verbose()
        
        task_id = self._cur_task + 1
        self.data_manager = data_manager
        
        if self.args['use_s_l_reg']:
            # self.load_slot_checkpoint(task_id)
            self.slot_learner.incremental_train(data_manager)
            self._ensure_slot_network_device(self._device)
            self.slot_learner._network.eval()
        
        self.learner.incremental_train(data_manager)
    
    
def cross_entropy_with_soft_labels(logits, soft_targets, normalized=False):
    """
    Calculate the cross-entropy loss for soft labels.

    Args:
        logits: Raw, unnormalized scores output from the model (shape: [batch_size, num_classes]).
        soft_targets: Probability distributions over classes (soft labels) (shape: [batch_size, num_classes]).

    Returns:
        The mean cross-entropy loss with soft labels.
    """
    if not normalized:
        # Apply log softmax to logits to get the log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
    else:
        log_probs = torch.log(logits)
    # log_soft_targets = torch.log(soft_targets)

    # Calculate the KL divergence loss
    loss = F.kl_div(log_probs, soft_targets, reduction='batchmean')     # the M-proj argmin_q KL(p||q), p: target; q: pred
    # loss = F.kl_div(log_probs, log_soft_targets, reduction='batchmean', log_target=True)

    return loss

