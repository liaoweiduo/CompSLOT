# CPrompt: https://github.com/Zhanxin-Gao/CPrompt

import logging
import copy
import numpy as np
import random
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import CPromptNet
from utils.toolkit import target2onehot, tensor2numpy, accuracy
from scipy.spatial.distance import cdist
from utils.toolkit import count_parameters
from models.base import BaseLearner
import os
from scipy import stats

num_workers = 8


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        
        self.args=args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._device = args['device'][0]
        self.dataset_name=args["dataset"]
        # self.args["num_classes"] = dataset_classes.get(self.dataset_name, 0) 
        self._network=CPromptNet(self.args, True)
        
        if not os.path.exists(os.path.join(args['logs_name'], 'checkpoints')):
            os.makedirs(os.path.join(args['logs_name'], 'checkpoints'))
        
    def after_task(self):
        self._known_classes = self._total_classes

    def _load_model(self, filename, drop_last=False):
        logging.info(f'=> Load from {filename}')
        state_dict = torch.load(filename)
        # complete with/without module.
        for key in list(state_dict.keys()):
            if 'module' in key:
                state_dict[key[7:]] = state_dict.pop(key)
            if drop_last and 'clas_w' in key:
                del state_dict[key]
        self._network.load_state_dict(state_dict, strict=True)
        logging.info(f'=> Load Done with params: {list(state_dict.keys())}')

        self._network.eval()

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1

        cur_task_nbclasses=data_manager.get_task_size(self._cur_task)
        self._total_classes = self._known_classes + cur_task_nbclasses
        self._network.update_fc(self._total_classes,cur_task_nbclasses)
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        
        m_rate = self.args['m_rate'] if 'm_rate' in self.args.keys() else 0.7 if 'comp_test' in self.args.keys() else None
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=None, m_rate=m_rate)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=8, persistent_workers=True, pin_memory=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=8)

        self._network.to(self._device)
        # pass training if has model
        need_train = True
        # try:
        #     checkpoint_name = os.path.join(self.args['logs_name'], 'checkpoints', f"{self.args['logfilename']}_{self._cur_task}.pkl")
        #     self._load_model(filename=checkpoint_name)
        #     need_train = False
        # except:
        #     logging.info(f'Do not find learned model, need to learn.')

        if need_train:
            
            if len(self._multiple_gpus) > 1:
                logging.info("Multiple GPUs")
                self._network = nn.DataParallel(self._network, self._multiple_gpus)
            self._train(self.train_loader, self.test_loader)
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module

            # save model
            # self._save_model(model_save_dir)
            self.save_checkpoint(os.path.join(self.args['logs_name'], 'checkpoints', 
                                            self.args['logfilename']))

        self._network.fix_branch_layer()
        
    def _train(self,train_loader,test_loader):
        enabled = set()
        enabled_params = []
        for name, param in self._network.named_parameters():
            if param.requires_grad:
                enabled.add(name)
                enabled_params.append(param)
        print(f"Parameters to be updated: {enabled}")

        # optimizer = optim.SGD(filter(lambda p: p.requires_grad, self._network.parameters()), momentum=0.9,lr=self.args["lr"],weight_decay=self.args["weight_decay"])
        optimizer = optim.SGD(enabled_params, momentum=0.9, lr=self.args["init_lr"], weight_decay=self.args["weight_decay"])

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.args["tuned_epoch"])

        self._classifier_train(train_loader,test_loader,optimizer,scheduler)

    def _get_loss(self, inputs, targets): 
        new_targets=targets-self._known_classes
        logits,features = self._network.aux_forward(inputs)
        # logging.info(f"logits {logits.shape}: {logits[0].detach().cpu().numpy()}")
        # logging.info(f"targets {new_targets.shape}: {new_targets}")
        loss_aux=F.cross_entropy(logits,new_targets)
        loss=loss_aux
        
        if self._cur_task>0:
            for k in range(self._cur_task):
                old_logit=self._network.clas_w[k](features)['logits']
                c1_logits=self._network.clas_w[self._cur_task](features)['logits']
                bool_=torch.max(c1_logits,dim=1)[0]>torch.max(old_logit,dim=1)[0]+self.args["margin"]
                t=torch.ones((bool_.shape)).to(self._device)
                t[bool_==False]=self.args["tau"]
                t=t.unsqueeze(1).repeat(1,old_logit.shape[1])
                # t=t.unsqueeze(1).repeat(1,self.args["increment"])
                ground=F.softmax(old_logit/t,dim=1).detach().clone()
                loss_ccl = -torch.sum(ground * torch.log(F.softmax(old_logit,dim=1)), dim=1).mean()
                loss+=self.args["alpha"]*loss_ccl/self._cur_task
                
        gen_p=[]
        x_querry = self._network.image_encoder(inputs, returnbeforepool=True)[:,0,:]
        K=self._network.keys

        s, f = self._known_classes, self._total_classes
        # s=self._cur_task*self.args["increment"]
        # f=(self._cur_task+1)*self.args["increment"]
        if self._cur_task==0:
            K = K[s:f]
        else:
            K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
        n_K = nn.functional.normalize(K, dim=1)
        q = nn.functional.normalize(x_querry, dim=1)
        mk = torch.einsum('bd,kd->bk', q, n_K)      # 只有10维

        # logging.info(f"mk {mk.shape}: {mk[0].detach().cpu().numpy()}")
        # logging.info(f"targets {targets.shape}: {targets}")

        loss_mk=F.cross_entropy(mk,targets)
        loss+=loss_mk
        
        m=torch.randint(0,self._cur_task+1,(len(mk),1))     # random select prompt for each sample

        ts_prompts_1=self._network.ts_prompts_1
        P1=torch.cat([ts_prompts_1[j].weight.unsqueeze(0) for j in m],dim=0)
        gen_p.append(P1)
        ts_prompts_2=self._network.ts_prompts_2
        P2=torch.cat([ts_prompts_2[j].weight.unsqueeze(0) for j in m],dim=0)
        gen_p.append(P2)
        out_gen=self._network(inputs,gen_p,train=True)
        loss_ce=F.cross_entropy(out_gen,new_targets)
        loss+=loss_ce

        return loss, {'logits': logits, 'new_targets': new_targets, 'features': features}
    
    def _classifier_train(self,train_loader,test_loader,optimizer,scheduler):
        prog_bar = tqdm(range(self.args["tuned_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                
                loss, res = self._get_loss(inputs, targets)
                logits = res['logits']
                new_targets = res['new_targets']
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(new_targets.expand_as(preds)).cpu().sum()
                total += len(targets)
            
            scheduler.step()
            train_acc = np.around(tensor2numpy(correct)*100 / total, decimals=2)
            
            info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}'.format(
                self._cur_task, epoch+1, self.args["tuned_epoch"], losses/len(train_loader), train_acc)
            
            prog_bar.set_description(info)
        logging.info(info)

    def update_cfst_fc(self, loader):
        # update fc after task training
        # Using w_slots because slot model does not have a standard vit forward path.
        
        label_task_map = np.zeros(self._total_classes)
        _cur_cls_id = 0
        for task_id in range(self._cur_task + 1):
            num_cls = self.args['init_cls'] if task_id == 0 else self.args['increment']
            for _ in range(num_cls):
                label_task_map[_cur_cls_id] = task_id
                _cur_cls_id = _cur_cls_id + 1

        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)
            
            gen_p=[]
            with torch.no_grad():
                x_querry = self._network.image_encoder(inputs, returnbeforepool=True)[:,0,:]
            
            K=self._network.keys
            
            f=self._total_classes
            # f=(self._cur_task+1)*self.args["increment"]
            K = K[:f]
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1)
            mk = torch.einsum('bd,kd->bk', q, n_K)      # the predict label for each sample

            if self._cur_task == 0:
                m=torch.max(mk,dim=1,keepdim=True)[1]//self.args["init_cls"]        # all 0 [b, 1]
            else:
                m = torch.zeros(mk.shape[0], 1).long().to(mk.device)
                for idx in range(mk.shape[0]):
                    task_id = label_task_map[torch.max(mk[idx], dim=0)[1].item()]
                    m[idx, 0] = int(task_id)
                # m=torch.max(mk,dim=1,keepdim=True)[1]//self.args["increment"]

            ts_prompts_1=self._network.ts_prompts_1
            P1=torch.cat([ts_prompts_1[j].weight.detach().clone().unsqueeze(0) for j in m],dim=0)
            gen_p.append(P1)
            ts_prompts_2=self._network.ts_prompts_2
            P2=torch.cat([ts_prompts_2[j].weight.detach().clone().unsqueeze(0) for j in m],dim=0)
            gen_p.append(P2)
            
            with torch.no_grad():
                _, features = self._network(inputs,gen_p,train=False, pen=True)
                fc = self.cfst_fc
                fc.set_support(features, targets)
                  
        logging.debug(f"Final support features and labels: {fc.support_labels}")
        logging.debug(f"Support features: {fc.support_features.shape}")
        logging.debug(f"Counter: {fc.label_feature_counter.feature_counts}")
        
    def _eval_cnn(self, loader, aux=False):
        self._network.to(self._device)
        
        label_task_map = np.zeros(self._total_classes)
        _cur_cls_id = 0
        for task_id in range(self._cur_task + 1):
            num_cls = self.args['init_cls'] if task_id == 0 else self.args['increment']
            for _ in range(num_cls):
                label_task_map[_cur_cls_id] = task_id
                _cur_cls_id = _cur_cls_id + 1

        self._network.eval()
        y_pred, y_true, recon_losses_pred = [], [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            
            gen_p=[]
            with torch.no_grad():
                x_querry = self._network.image_encoder(inputs, returnbeforepool=True)[:,0,:]
            
            K=self._network.keys
            
            f=self._total_classes
            # f=(self._cur_task+1)*self.args["increment"]
            K = K[:f]
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1)
            mk = torch.einsum('bd,kd->bk', q, n_K)      # the predict label for each sample

            if self._cur_task == 0:
                m=torch.max(mk,dim=1,keepdim=True)[1]//self.args["init_cls"]        # all 0 [b, 1]
            else:
                m = torch.zeros(mk.shape[0], 1).long().to(mk.device)
                for idx in range(mk.shape[0]):
                    task_id = label_task_map[torch.max(mk[idx], dim=0)[1].item()]
                    m[idx, 0] = int(task_id)
                # m=torch.max(mk,dim=1,keepdim=True)[1]//self.args["increment"]

            # if self.args['debug']:
            #     logging.info(f'DEBUG: mk {mk.shape}: {mk[0]}')
            #     logging.info(f'DEBUG: m {m.shape}: {m[0]}')

            ts_prompts_1=self._network.ts_prompts_1
            P1=torch.cat([ts_prompts_1[j].weight.detach().clone().unsqueeze(0) for j in m],dim=0)
            gen_p.append(P1)
            ts_prompts_2=self._network.ts_prompts_2
            P2=torch.cat([ts_prompts_2[j].weight.detach().clone().unsqueeze(0) for j in m],dim=0)
            gen_p.append(P2)
            
            with torch.no_grad():
                if not aux:
                    out_logits=self._network(inputs,gen_p,train=False)
                else:
                    _, features = self._network(inputs,gen_p,train=False, pen=True)
                    out_logits = self.cfst_fc(features)['logits']
                    
            predicts = torch.topk(out_logits, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def normal_eval_cnn(self,loader):
        self._network.eval()
        
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                logits = self._network(inputs)
                
            predicts = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
