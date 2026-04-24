import os
import ast
import math
import logging
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import (iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, iCIFAR224, 
                        iImageNetR,iImageNetA,CUB, objectnet, omnibenchmark, vtab, 
                        iCGQA, iCOBJ)


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args):
        self.args = args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)
        
        if 'cfst' in args['dataset'].lower():       # cfst_cgqa, cfst_cobj
            self.num_cfst_exps = 300
            self.n_way, self.n_shot, self.n_query = 10, 10, 10, 
            self._setup_cfst_data()
    
    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    @property
    def nb_classes(self):
        return len(self._class_order)

    def get_transforms(self, mode): 
        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode in ["test", "val"]:
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))
    
        return trsf
    
    def get_dataset_cfst(
        self, task_id, cfst_mode, mode='test', ret_data=False, n_shot=None, n_query=None
    ):
        trsf = self.get_transforms(mode)
        
        if n_shot is None: 
            n_shot = self.n_shot
        if n_query is None: 
            n_query = self.n_query
        class_order = self._cfst_class_order[cfst_mode][task_id]
        
        # select (n_shot+n_query)*n_way samples
        sup_data, sup_targets = [], []
        que_data, que_targets = [], []
        rng = self.cfst_rng
        for re_cls_idx, cls_idx in enumerate(class_order): 
            class_data = self.cfst_class_data[cfst_mode][cls_idx]
            # class_targets = self.cfst_class_targets[cfst_mode][cls_idx]
            indices = rng.choice(len(class_data), n_shot + n_query, replace=False)
            sup_data.append(class_data[indices[:n_shot]])
            # sup_targets.append(class_targets[indices[:n_shot]])           # true label
            sup_targets.append(np.ones(n_shot).astype(int)*re_cls_idx)      # relative label
            que_data.append(class_data[indices[n_shot:]])
            # que_targets.append(class_targets[indices[n_shot:]])           # true label
            que_targets.append(np.ones(n_query).astype(int)*re_cls_idx)     # relative label
        sup_data = np.concatenate(sup_data)
        sup_targets = np.concatenate(sup_targets)
        que_data = np.concatenate(que_data)
        que_targets = np.concatenate(que_targets)
        
        logging.debug(f'Get cfst data: {cfst_mode}, task {task_id}, {n_shot}-shot {n_query}-query {len(class_order)}-way.')
        logging.debug(f'sup_data {sup_data.shape}, que_data {que_data.shape}')
        
        if ret_data:
            return (sup_data, sup_targets, que_data, que_targets, 
                    DummyDataset(sup_data, sup_targets, trsf, use_path=self.use_path, args=self.args, mode=mode), 
                    DummyDataset(que_data, que_targets, trsf, use_path=self.use_path, args=self.args, mode=mode))
        else:
            return (DummyDataset(sup_data, sup_targets, trsf, use_path=self.use_path, args=self.args, mode=mode), 
                    DummyDataset(que_data, que_targets, trsf, use_path=self.use_path, args=self.args, mode=mode))

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "val":
            x, y = self._val_data, self._val_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        trsf = self.get_transforms(mode)
        
        data, targets = [], []
        
        if len(x) > 0:
            for idx in indices:
                if m_rate is None:
                    class_data, class_targets = self._select(
                        x, y, low_range=idx, high_range=idx + 1)
                else:
                    class_data, class_targets = self._select_rmm(
                        x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate)
                data.append(class_data)
                targets.append(class_targets)
                
            if appendent is not None and len(appendent) != 0:
                appendent_data, appendent_targets = appendent
                
                data.append(appendent_data)
                targets.append(appendent_targets)

            data, targets = np.concatenate(data), np.concatenate(targets)
            
            dataset = DummyDataset(data, targets, trsf, use_path=self.use_path, args=self.args, mode=mode, idata=self.idata)
        
        else: 
            dataset = None
        
        if ret_data:
            return data, targets, dataset
        else:
            return dataset

    def get_dataset_with_split(
        self, indices, source, mode, appendent=None, val_samples_per_class=0
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        elif source == "val":
            x, y = self._val_data, self._val_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(
                x, y, low_range=idx, high_range=idx + 1
            )
            val_indx = np.random.choice(
                len(class_data), val_samples_per_class, replace=False
            )
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(
                    len(append_data), val_samples_per_class, replace=False
                )
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(
            train_targets
        )
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(train_data, train_targets, trsf, use_path=self.use_path, args=self.args, mode=mode
        ), DummyDataset(val_data, val_targets, trsf, use_path=self.use_path, args=self.args, mode=mode)

    def _setup_cfst_data(self):
        # Generate cfst exps with a order of classes and data for each target
        seed = 42
        self.cfst_rng = np.random.RandomState(seed=seed)
        self.cfst_modes = self.idata.cfst_modes
        self._cfst_class_order = {}
        self.cfst_class_data, self.cfst_class_targets = {}, {}
        for mode in self.cfst_modes: 
            # Generate class order for each exp
            class_order = np.unique(self.idata.cfst_targets[mode]).tolist()
            self._cfst_class_order[mode] = []
            for exp_idx in range(self.num_cfst_exps):
                '''select n_way classes for each exp'''
                selected_class_idxs = self.cfst_rng.choice(class_order, self.n_way, replace=False).astype(np.int64).tolist()
                self._cfst_class_order[mode].append(selected_class_idxs)
            # logging.debug(f"CFST mode {mode}: Class order: \n{self._cfst_class_order[mode]}") 

            # Split class-wise data and targets
            self.cfst_class_data[mode], self.cfst_class_targets[mode] = {}, {}
            for idx in class_order: 
                class_data, class_targets = self._select(
                    self.idata.cfst_data[mode], self.idata.cfst_targets[mode], low_range=idx, high_range=idx + 1
                )
                self.cfst_class_data[mode][idx] = class_data
                self.cfst_class_targets[mode][idx] = class_targets
                
    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()
        self.idata = idata

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        try:
            self._val_data, self._val_targets = idata.val_data, idata.val_targets
        except:
            self._val_data, self._val_targets = None, None
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path
        
        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf
        self.norm = idata.norm

        # Order
        order = [i for i in range(len(np.unique(self._test_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(f'Class order: {self._class_order}')

        # Map indices
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)
        
    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))

        
class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False, 
                 args=None, mode='test', idata=None):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path
        self.args = args
        self.mode = mode
        self.idata = idata
        
    def __len__(self):
        return len(self.images)

    def get_image_by_path(self, image_path):
        image = pil_loader(image_path)
        image = self.trsf(image)
        return image

    def __getitem__(self, idx):
        if self.use_path:
            # Handle issue while loading images in a parallel run.
            success = False
            count = 0
            while not success:
                try:
                    img_path = self.images[idx]
                    # only contain path
                    image = pil_loader(img_path)
                    
                    success = True
                except Exception as e:
                    count += 1
                    if count > 50:
                        raise RuntimeError(f"Failed to load image after 10 attempts. Last index tried: id {idx}; {img_path}.")
                    idx_old = idx
                    idx = np.random.randint(0, len(self.images))
                    logging.debug(f"Failed to load image at index {idx_old}: {e}. Trying another index {idx}.")
            
            # logging.debug(f"Image path: {self.images[idx]}, width: {image.width}, height: {image.height}")
            image = self.trsf(image)
            # logging.debug(f"Loaded image at index {idx} successfully. image: {image.shape}")
            
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
                
        label = self.labels[idx]

        return idx, image, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    elif name == "cifar224":
        return iCIFAR224(args)
    elif name == "imagenetr":
        return iImageNetR(args)
    elif name == "imageneta":
        return iImageNetA()
    elif name == "cub":
        return CUB()
    elif name == "objectnet":
        return objectnet()
    elif name == "omnibenchmark":
        return omnibenchmark()
    elif name == "vtab":
        return vtab()
    elif name == "cfst_cgqa": 
        return iCGQA()
    elif name == "cfst_cobj":
        return iCOBJ()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    return pil_loader(path)
