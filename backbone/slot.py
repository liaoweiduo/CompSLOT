import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class SlotAttention(nn.Module):
    def __init__(self, args, num_patches):
        super().__init__()
        self.task_count = 0
        if "large" in args["backbone_type"]: 
            self.emb_d = 1024
        else:
            self.emb_d = 768
        self.key_d = args["slot_dim"]
        self.num_patches = num_patches
        logging.debug("key_d: {}, num_patches: {}".format(
            self.key_d, self.num_patches))
        
        # slot basic param
        self.n_slots = args["n_slots"]
        self.n_iter = args["n_iter"]
        self.temp = args["slot_temp"]
        self.attn_epsilon = 1e-8
        self.gru_d = self.key_d
        logging.debug("n_slots: {}, n_iter: {}, temp: {}".format(self.n_slots, self.n_iter, self.temp))

        # slot encoder
        self.ln_input = nn.LayerNorm(self.emb_d)
        self.ln_slot = nn.LayerNorm(self.key_d)
        self.ln_output = nn.LayerNorm(self.key_d)
        self.mu = self.init_tensor(1, 1, self.key_d)
        self.log_sigma = self.init_tensor(1, 1, self.key_d)
        self.k = nn.Linear(self.emb_d, self.key_d, bias=False)
        self.q = nn.Linear(self.key_d, self.key_d, bias=False)
        self.v = nn.Linear(self.emb_d, self.key_d, bias=False)
        
        self.gru = nn.GRUCell(self.key_d, self.gru_d)
        self.mlp = nn.Sequential(
            nn.Linear(self.key_d, self.key_d, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.key_d, self.key_d, bias=True)
        )
        
        # slot decoder
        self.decoder_pos_emb = nn.Parameter(torch.zeros(1, num_patches, 1, self.key_d))
        trunc_normal_(self.decoder_pos_emb, std=.02)
        self.decoder_mapping_layers = args.get('decoder_mapping_layers', 3)
        
        decoder_dim = self.key_d * 2
        decoder_layers = [
            nn.Linear(self.key_d, decoder_dim, bias=True),
        ]
        for _ in range(self.decoder_mapping_layers - 2):
            decoder_layers.append(nn.ReLU(inplace=True))    
            decoder_layers.append(nn.Linear(decoder_dim, decoder_dim, bias=True))
        decoder_layers.append(nn.ReLU(inplace=True))
        decoder_layers.append(nn.Linear(decoder_dim, self.emb_d, bias=True))
        self.decoder = nn.Sequential(*decoder_layers)

        # primitive selector param
        self.select_slot_temp = args["select_slot_temp"]
        logging.debug("select_slot_temp: {}".format(self.select_slot_temp))
        
        # primitive selector
        self.slot_ln = nn.LayerNorm(self.key_d) if not args.get('disable_ln', False) else nn.Identity()
        self.task_key = nn.Parameter(torch.randn(self.key_d), requires_grad=True)
        self.slot_selection_w = self.init_tensor(self.key_d, self.key_d)
        self.slot_selection_b = self.init_tensor(self.key_d)
        
        self.debug_verbose = 1
    
    def process_task_count(self):
        self.task_count += 1
    
    def forward(self, features, temp=None, n_iter=None):
        """forward all path
        """
        slots, attn, _ = self.forward_slots(features, temp=temp, n_iter=n_iter)

        # recon
        slot_features = slots.unsqueeze(1)
        slot_features = slot_features + self.decoder_pos_emb
        slot_features = self.decoder(slot_features)
        slot_features = torch.einsum('bnkd,bnk->bnd', slot_features, attn)

        # recon loss
        recon_loss = F.mse_loss(slot_features, features, reduction='none')
        recon_loss = torch.mean(torch.mean(recon_loss, dim=-1), dim=-1)

        # primitive selection
        w, w_slots = self.forward_selector(slots)
        
        if self.debug_verbose > 0:
            logging.debug("slots: {} {}".format(slots.shape, slots[0,0,:10]))
            logging.debug("attn: {}".format(attn.shape))
            logging.debug("slot_features: {}".format(slot_features.shape))
            logging.debug("recon_loss: {} {}".format(recon_loss.shape, recon_loss[0]))
            logging.debug("w: {} {}".format(w.shape, w[0]))
            logging.debug("w_slots: {}".format(w_slots.shape))
            self.debug_verbose -= 1
        
        return {'slots': slots, 'attns': attn, "recon_loss": recon_loss, "w": w, "w_slots": w_slots}

    def forward_slots(self, features, temp=None, n_iter=None):
        bs = features.shape[0]

        n_iter = self.n_iter if n_iter is None else n_iter
        temp = self.temp if temp is None else temp
        iter_slots = []
        iter_attn_vis = []

        # init
        features = self.ln_input(features)
        slots = torch.randn(
            bs, self.n_slots, self.key_d, device=self.log_sigma.device
            ) * torch.exp(self.log_sigma) + self.mu

        # iter
        k = self.k(features)
        v = self.v(features)
        k = (self.key_d ** (-0.5) * temp) * k

        attn_vis = None
        for t in range(n_iter):
            slots_prev = slots.clone()
            slots = self.ln_slot(slots)
            q = self.q(slots)

            # b = bs, n = 196, k = 5, d = 64
            ## softmax(KQ^T/sqrt(d), dim='slots')
            # sum((b x n x 1 x d) * [b x 1 x k x d]) = (b x n x k)
            attn = torch.einsum('bnd,bkd->bnk', k, q)
            # attn = attn * (self.key_d ** -0.5)
            # softmax over slots
            attn_vis = F.softmax(attn, dim=-1)      # [b, n, k]

            ## updates = WeightedMean(attn+epsilon, v)
            attn = attn_vis + self.attn_epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            # sum((b x n x k x 1) * (b x n x 1 x d)) = (b x k x d)
            updates = torch.einsum('bnk,bnd->bkd', attn, v)

            ## slots = GRU(state=slots_prev[b,k,d], inputs=updates[b,k,d])  (for each slot)
            slots = self.gru(updates.view(-1, self.key_d),               # [b*k, d]
                             slots_prev.reshape(-1, self.key_d))         # [b*k, d]
            slots = slots.view(bs, self.n_slots, self.key_d)        # [b, k, d]

            ## slots += MLP(LayerNorm(slots))
            slots = slots + self.mlp(self.ln_output(slots))

            iter_slots.append(slots.detach().clone())
            iter_attn_vis.append(attn_vis.detach().clone())

        return slots, attn_vis, {'slots': iter_slots, 'attns': iter_attn_vis}
    
    def forward_selector(self, slots):
        """aggregate slots

        Args:
            slots (Tensor): [bs, n, d]
        """
        bs, n, h = slots.shape
        
        w = torch.ones(bs, n).to(slots.device)      # default weights if not used
        w_slots = None
        slots = self.slot_ln(slots)  # apply layernorm to alleviate shifting in slots

        slot_selection_w = self.slot_selection_w
        slot_selection_b = self.slot_selection_b
        # [bs, n10, h128] @ [h128, d128] -> [bs, n10, d128]
        mapped_slots = torch.einsum('bnh,hd->bnd', slots, slot_selection_w)
        mapped_slots = mapped_slots + slot_selection_b
        # mapped_slots = self.slot_ln2(mapped_slots)
        mapped_slots = torch.tanh(mapped_slots)
        task_key = self.task_key  # [128] or [self.n_tasks, 128]
        
        # softmax(1/sqrt(D) S_m@K_t)
        w = torch.einsum('bnd,d->bn', mapped_slots, task_key)
        w = w * (task_key.shape[-1] ** -0.5)
        w = w * self.select_slot_temp
        w = F.softmax(w, dim=-1)
        w_slots = torch.einsum('bnh,bn->bh', mapped_slots, w)  # weighted slots
        
        # w: [bs, n10]
        return w, w_slots

    def init_tensor(self, a, b=None, c=None, d=None, init=True, ortho=False):
        if b is None:
            p = torch.nn.Parameter(torch.FloatTensor(a), requires_grad=True)
        elif c is None:
            p = torch.nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
        elif d is None:
            p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
        else:
            p = torch.nn.Parameter(torch.FloatTensor(a, b, c, d), requires_grad=True)
        if init:
            if ortho:
                nn.init.orthogonal_(p)
            elif b is None:         # for bias
                nn.init.constant_(p, 0)
            else:               # for weight
                nn.init.xavier_uniform_(p)
        return p
