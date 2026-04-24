import os
import logging

import os.path
from pathlib import Path

import json

import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm import tqdm
from datetime import datetime

from PIL import Image


class CGQA(torch.utils.data.Dataset):
    """Load train, val test, and all CFST subsets"""
    def __init__(self, root, download=False):
        self.root = os.path.expanduser(root)

        self.meta_info = dict()
        # load continual set
        datasets, meta_info = self._get_datasets(
            self.root, mode="continual", image_size=(224, 224),
            load_set='all' if download else None)
        self.train_set, self.val_set, self.test_set = datasets['train'], datasets['val'], datasets['test']
        self.meta_info['continual'] = meta_info
        
        self.classes = np.unique(self.test_set.targets)         # [0->99]
        
        self.cfst_set = {}
        self._get_cfst_set()
    
    def _get_cfst_set(self):        # different from COBJ
        # load cfst test set
        for mode in ['sys', 'pro', 'sub', 'non', 'noc']: 
            datasets, meta_info = self._get_datasets(
                self.root, mode=mode, image_size=(224, 224))
            self.cfst_set[mode] = datasets['dataset']
            self.meta_info[mode] = meta_info
            
    def __getitem__(self, index):
        raise NotImplementedError('Do not directly use this method to load samples')
        if self.train:
            data = self.train_set[index]
        else:
            data = self.test_set[index]

        img, target, ori_idx = data

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _get_datasets(self, dataset_root, image_size=(128, 128), shuffle=False, seed=None,
                      mode='continual', num_samples_each_label=None, label_offset=0, load_set=None):
        """
        Create GQA dataset, with given json files,
        containing instance tuples with shape (img_name, label).

        You may need to specify label_offset if relative label do not start from 0.

        :param dataset_root: Path to the dataset root folder.
        :param image_size: size of image.
        :param shuffle: If true, the train sample order (in json)
            in the incremental experiences is
            randomly shuffled. Default to False.
        :param seed: A valid int used to initialize the random number generator.
            Can be None.
        :param mode: Option [continual, sys, pro, sub, non, noc, nons, syss].
        :param num_samples_each_label: If specify a certain number of samples for each label,
            random sampling (build-in seed:1234,
            and replace=True if num_samples_each_label > num_samples, else False)
            is used to sample.
            Only for continual mode, only apply to train dataset.
        :param label_offset: specified if relative label not start from 0.
        :param load_set: all -> load all set;
            train -> only pre-load train set;
            val -> only pre-load val set;
            test -> only pre-load test set.
            default None.

        :return data_sets defined by json file and label information.
        """
        img_folder_path = os.path.join(dataset_root, "CFST", "CGQA", "GQA_100")

        def preprocess_concept_to_integer(img_info, mapping_tuple_label_to_int_concepts):
            for item in img_info:
                item['concepts'] = [mapping_tuple_label_to_int_concepts[concept] for concept in item['comb']]

        def preprocess_label_to_integer(img_info, mapping_tuple_label_to_int):
            for item in img_info:
                item['image'] = f"{item['newImageName']}.jpg"
                item['label'] = mapping_tuple_label_to_int[tuple(sorted(item['comb']))]
                for obj in item['objects']:
                    obj['image'] = f"{obj['imageName']}.jpg"

        def formulate_img_tuples(images):
            """generate train_list and test_list: list with img tuple (path, label)"""
            img_tuples = []
            for item in images:
                instance_tuple = (
                item['image'], item['label'], item['concepts'], item['position'])  # , item['boundingBox']
                img_tuples.append(instance_tuple)
            return img_tuples

        if mode == 'continual':
            train_json_path = os.path.join(img_folder_path, "continual", "train", "train.json")
            val_json_path = os.path.join(img_folder_path, "continual", "val", "val.json")
            test_json_path = os.path.join(img_folder_path, "continual", "test", "test.json")

            with open(train_json_path, 'r') as f:
                train_img_info = json.load(f)
            with open(val_json_path, 'r') as f:
                val_img_info = json.load(f)
            with open(test_json_path, 'r') as f:
                test_img_info = json.load(f)
            # img_info:
            # [{'newImageName': 'continual/val/59767',
            #   'comb': ['hat', 'leaves'],
            #   'objects': [{'imageName': '2416370', 'objName': 'hat',
            #                'attributes': ['red'], 'boundingBox': [52, 289, 34, 45]},...]
            #   'position': [4, 1]},...]

            '''preprocess labels to integers'''
            label_set = sorted(list(set([tuple(sorted(item['comb'])) for item in val_img_info])))
            # [('building', 'sign'), ...]
            map_tuple_label_to_int = dict((item, idx + label_offset) for idx, item in enumerate(label_set))
            # {('building', 'sign'): 0, ('building', 'sky'): 1, ...}
            map_int_label_to_tuple = dict((idx + label_offset, item) for idx, item in enumerate(label_set))
            # {0: ('building', 'sign'), 1: ('building', 'sky'),...}
            '''preprocess concepts to integers'''
            concept_set = sorted(list(set([concept for item in val_img_info for concept in item['comb']])))
            mapping_tuple_label_to_int_concepts = dict((item, idx) for idx, item in enumerate(concept_set))
            # 21 concepts {'bench': 0, 'building': 1, 'car': 2, ...}
            map_int_concepts_label_to_str = dict((idx, item) for idx, item in enumerate(concept_set))
            # 21 concepts {0: 'bench', 1: 'building', 2: 'car', ...}
            map_int_label_to_concept = dict(
                (label, [mapping_tuple_label_to_int_concepts[concept_str] for concept_str in tuple_concepts]) 
                for label, tuple_concepts in map_int_label_to_tuple.items())
            # {0: [1, 10], ...}
            
            preprocess_label_to_integer(train_img_info, map_tuple_label_to_int)
            preprocess_label_to_integer(val_img_info, map_tuple_label_to_int)
            preprocess_label_to_integer(test_img_info, map_tuple_label_to_int)

            preprocess_concept_to_integer(train_img_info, mapping_tuple_label_to_int_concepts)
            preprocess_concept_to_integer(val_img_info, mapping_tuple_label_to_int_concepts)
            preprocess_concept_to_integer(test_img_info, mapping_tuple_label_to_int_concepts)

            '''if num_samples_each_label provided, sample images to balance each class for train set'''
            selected_train_images = []
            if num_samples_each_label is not None and num_samples_each_label > 0:
                imgs_each_label = dict()
                for item in train_img_info:
                    label = item['label']
                    if label in imgs_each_label:
                        imgs_each_label[label].append(item)
                    else:
                        imgs_each_label[label] = [item]
                build_in_seed = 1234
                build_in_rng = np.random.RandomState(seed=build_in_seed)
                for label, imgs in imgs_each_label.items():
                    selected_idxs = build_in_rng.choice(
                        np.arange(len(imgs)), num_samples_each_label,
                        replace=True if num_samples_each_label > len(imgs) else False)
                    for idx in selected_idxs:
                        selected_train_images.append(imgs[idx])
            else:
                selected_train_images = train_img_info

            '''generate train_list and test_list: list with img tuple (path, label)'''
            train_list = formulate_img_tuples(selected_train_images)
            val_list = formulate_img_tuples(val_img_info)
            test_list = formulate_img_tuples(test_img_info)
            # [('continual/val/59767.jpg', 0),...

            '''shuffle the train set'''
            if shuffle:
                rng = np.random.RandomState(seed=seed)
                order = np.arange(len(train_list))
                rng.shuffle(order)
                train_list = [train_list[idx] for idx in order]

            '''generate train_set and test_set using PathsDataset'''
            train_set = self.PathsDataset(
                root=img_folder_path,
                files=train_list,  # train_list,      val_list for debug
                transform=transforms.Compose([transforms.Resize(image_size)]),
                loaded=load_set in ['train', 'all'],
                name='con_train',
            )
            val_set = self.PathsDataset(
                root=img_folder_path,
                files=val_list,
                transform=transforms.Compose([transforms.Resize(image_size)]),
                loaded=load_set in ['val', 'all'],
                name='con_val',
            )
            test_set = self.PathsDataset(
                root=img_folder_path,
                files=test_list,
                transform=transforms.Compose([transforms.Resize(image_size)]),
                loaded=load_set in ['test', 'all'],
                name='con_test',
            )

            datasets = {'train': train_set, 'val': val_set, 'test': test_set}
            meta_info = {
                "label_set": label_set, 
                "map_tuple_label_to_int": map_tuple_label_to_int, 
                "map_int_label_to_tuple": map_int_label_to_tuple, 
                "concept_set": concept_set,
                "mapping_tuple_label_to_int_concepts": mapping_tuple_label_to_int_concepts,
                "map_int_concepts_label_to_str": map_int_concepts_label_to_str,
                "map_int_label_to_concept": map_int_label_to_concept, 
                "train_list": train_list, "val_list": val_list, "test_list": test_list, 
                "img_folder_path": img_folder_path}

        elif mode in ['sys', 'pro', 'sub', 'non', 'noc']:
            json_name = \
            {'sys': 'sys/sys_fewshot.json', 'pro': 'pro/pro_fewshot.json', 'sub': 'sub/sub_fewshot.json',
             'non': 'non_novel/non_novel_fewshot.json', 'noc': 'non_comp/non_comp_fewshot.json'}[mode]
            json_path = os.path.join(img_folder_path, "fewshot", json_name)
            with open(json_path, 'r') as f:
                img_info = json.load(f)
            label_set = sorted(list(set([tuple(sorted(item['comb'])) for item in img_info])))
            map_tuple_label_to_int = dict((item, idx + label_offset) for idx, item in enumerate(label_set))
            map_int_label_to_tuple = dict((idx + label_offset, item) for idx, item in enumerate(label_set))
            preprocess_label_to_integer(img_info, map_tuple_label_to_int)
            concept_set = sorted(list(set([concept for item in img_info for concept in item['comb']])))
            mapping_tuple_label_to_int_concepts = dict((item, idx) for idx, item in enumerate(concept_set))
            map_int_concepts_label_to_str = dict((idx, item) for idx, item in enumerate(concept_set))
            preprocess_concept_to_integer(img_info, mapping_tuple_label_to_int_concepts)
            # map_int_label_to_concept = dict()
            # for item in img_info:
            #     if item['label'] not in map_int_label_to_concept.keys():
            #         map_int_label_to_concept[item['label']] = item['concepts']
            map_int_label_to_concept = dict(
                (label, [mapping_tuple_label_to_int_concepts[concept_str] for concept_str in tuple_concepts]) 
                for label, tuple_concepts in map_int_label_to_tuple.items())
            # {0: [1, 10], ...}
            
            img_list = formulate_img_tuples(img_info)
            dataset = self.PathsDataset(
                root=img_folder_path,
                files=img_list,
                transform=transforms.Compose([transforms.Resize(image_size)]),
                loaded=True,
                name=f'few_{mode}',
            )

            datasets = {'dataset': dataset}
            meta_info = {
                "label_set": label_set, 
                "map_tuple_label_to_int": map_tuple_label_to_int, 
                "map_int_label_to_tuple": map_int_label_to_tuple, 
                "concept_set": concept_set,
                "mapping_tuple_label_to_int_concepts": mapping_tuple_label_to_int_concepts,
                "map_int_concepts_label_to_str": map_int_concepts_label_to_str,
                "map_int_label_to_concept": map_int_label_to_concept,
                "img_list": img_list,
                "img_folder_path": img_folder_path,}\

        else:
            raise Exception(f'Un-implemented mode "{mode}".')

        return datasets, meta_info

    class PathsDataset(torch.utils.data.Dataset):
        """
        This class extends the basic Pytorch Dataset class to handle list of paths
        as the main data source.
        """

        def __init__(
                self,
                root,
                files,
                transform=None,
                target_transform=None,
                loaded=True,
                name='data',
        ):
            """
            Creates a File Dataset from a list of files and labels.

            :param root: root path where the data to load are stored. May be None.
            :param files: list of tuples. Each tuple must contain two elements: the
                full path to the pattern and its class label. Optionally, the tuple
                may contain a third element describing the bounding box to use for
                cropping (top, left, height, width).
            :param transform: eventual transformation to add to the input data (x)
            :param target_transform: eventual transformation to add to the targets
                (y)
            :param loader: loader function to use (for the real data) given path.
            :param loaded: True, load images into memory.
            If False, load when call getitem.
            Default True.
            :param name: Name if save to folder
            """

            if root is not None:
                root = Path(root)

            self.root = root
            self.imgs = files
            self.targets = np.array([img_data[1] for img_data in self.imgs])
            self.concepts = [img_data[2] for img_data in self.imgs]  # [bs, 2]
            
            self.transform = transform
            self.target_transform = target_transform
            self.loaded = loaded
            self.name = name

            if self.loaded:
                self.load_data()

            try:
                self.data = np.stack([np.asarray(self.transform(img_data[0])) for img_data in self.imgs])     # only x
            except:
                logging.info(f"Error loadding data as array, load sample paths instead.")
                self.data = np.stack([self.root / img_data[0] for img_data in self.imgs])

        def load_data(self):
            """
            load all data and replace imgs.
            """
            logging.info(f'[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] Load data in PathsDataset.')

            # if has saved, just load
            if os.path.exists(os.path.join(self.root, f'{self.name}.npy')):
                data = np.load(os.path.join(self.root, f'{self.name}.npy'), allow_pickle=True).item()
                self.imgs = data['imgs']
                self.targets = np.array(data['targets'])
            else:
                for index in tqdm(range(len(self.imgs))):
                    impath = self.imgs[index][0]
                    if self.root is not None:
                        impath = self.root / impath
                    img = Image.open(impath).convert("RGB")

                    self.imgs[index] = (img, *self.imgs[index][1:])

                # save self.imgs and targets to root
                data = {'imgs': self.imgs, 'targets': self.targets}
                np.save(os.path.join(self.root, f'{self.name}.npy'), data)

            logging.info(f'[{datetime.now().strftime("%Y/%m/%d %H:%M:%S")}] DONE.')

        def __getitem__(self, index, return_concepts=False):
            """
            Returns next element in the dataset given the current index.

            :param index: index of the data to get.
            :return: loaded item.
            """

            img_description = self.imgs[index]
            impath = img_description[0]
            target = img_description[1]
            bbox = None
            concepts, position = None, None
            if len(img_description) == 3:
                concepts = img_description[2]
            elif len(img_description) == 4:
                concepts = img_description[2]
                position = img_description[3]

            if self.loaded:
                img = impath
            else:
                if self.root is not None:
                    impath = self.root / impath
                img = Image.open(impath).convert("RGB")

            # If a bounding box is provided, crop the image before passing it to
            # any user-defined transformation.
            if bbox is not None:
                if isinstance(bbox, torch.Tensor):
                    bbox = bbox.tolist()
                img = crop(img, *bbox)

            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                target = self.target_transform(target)

            # If provide concepts and position,
            if return_concepts:
                return img, target, index, concepts, position
            else:
                return img, target, index

        def __len__(self):
            """
            Returns the total number of elements in the dataset.

            :return: Total number of dataset items.
            """

            return len(self.imgs)


class COBJ(CGQA):
    # def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
    #     super().__init__(root, train, transform, target_transform, download)

    def _get_cfst_set(self): 
        # load cfst test set
        for mode in ['sys', 'pro', 'non', 'noc']:       # no 'sub'
            datasets, meta_info = self._get_datasets(
                self.root, mode=mode, image_size=(224, 224))
            self.cfst_set[mode] = datasets['dataset']
            self.meta_info[mode] = meta_info

    def _get_datasets(self, dataset_root, image_size=(128, 128), shuffle=False, seed=None,
                      mode='continual', num_samples_each_label=None, label_offset=0, load_set=None):

        """
        Create COBJ dataset, with given json files,
        containing instance tuples with shape (img_name, label).

        You may need to specify label_offset if relative label do not start from 0.

        :param dataset_root: Path to the dataset root folder.
        :param image_size: size of image.
        :param shuffle: If true, the train sample order (in json)
            in the incremental experiences is
            randomly shuffled. Default to False.
        :param seed: A valid int used to initialize the random number generator.
            Can be None.
        :param mode: Option [continual, sys, pro, sub, non, noc, nons, syss].
        :param num_samples_each_label: If specify a certain number of samples for each label,
            random sampling (build-in seed:1234,
            and replace=True if num_samples_each_label > num_samples, else False)
            is used to sample.
            Only for continual mode, only apply to train dataset.
        :param label_offset: specified if relative label not start from 0.
        :param load_set: train -> only pre-load train set;
            val -> only pre-load val set;
            test -> only pre-load test set.
            default None.

        :return data_sets defined by json file and label information.
        """
        img_folder_path = os.path.join(dataset_root, "CFST", "COBJ", "annotations")

        def preprocess_concept_to_integer(img_info, mapping_tuple_label_to_int_concepts):
            for item in img_info:
                item['concepts'] = [mapping_tuple_label_to_int_concepts[concept] for concept in (
                    item['label'] if isinstance(item['label'], list) else [item['label']])]

        def preprocess_label_to_integer(img_info, mapping_tuple_label_to_int, prefix=''):
            for item in img_info:
                item['image'] = f"{prefix}{item['imageId']}.jpg"
                item['label'] = mapping_tuple_label_to_int[tuple(sorted(item['label']))]

        def formulate_img_tuples(images):
            """generate train_list and test_list: list with img tuple (path, label)"""
            img_tuples = []
            for item in images:
                if 'concepts' in item.keys():
                    instance_tuple = (item['image'], item['label'], item['concepts'])  # , item['boundingBox']
                else:
                    instance_tuple = (item['image'], item['label'])
                img_tuples.append(instance_tuple)
            return img_tuples

        if mode == 'continual':
            train_json_path = os.path.join(img_folder_path, "O365_continual_train_crop.json")
            val_json_path = os.path.join(img_folder_path, "O365_continual_val_crop.json")
            test_json_path = os.path.join(img_folder_path, "O365_continual_test_crop.json")

            with open(train_json_path, 'r') as f:
                train_img_info = json.load(f)
            with open(val_json_path, 'r') as f:
                val_img_info = json.load(f)
            with open(test_json_path, 'r') as f:
                test_img_info = json.load(f)
            # img_info:
            # [{'newImageName': 'continual/val/59767',
            #   'comb': ['hat', 'leaves'],
            #   'objects': [{'imageName': '2416370', 'objName': 'hat',
            #                'attributes': ['red'], 'boundingBox': [52, 289, 34, 45]},...]
            #   'position': [4, 1]},...]

            '''preprocess labels to integers'''
            label_set = sorted(list(set([tuple(sorted(item['label'])) for item in val_img_info])))
            # [('building', 'sign'), ...]
            map_tuple_label_to_int = dict((item, idx + label_offset) for idx, item in enumerate(label_set))
            # {('building', 'sign'): 0, ('building', 'sky'): 1, ...}
            map_int_label_to_tuple = dict((idx + label_offset, item) for idx, item in enumerate(label_set))
            # {0: ('building', 'sign'), 1: ('building', 'sky'),...}
            '''preprocess concepts to integers'''
            concept_set = sorted(list(set([concept for item in val_img_info for concept in item['label']])))
            mapping_tuple_label_to_int_concepts = dict((item, idx) for idx, item in enumerate(concept_set))
            map_int_concepts_label_to_str = dict((idx, item) for idx, item in enumerate(concept_set))
            map_int_label_to_concept = dict(
                (label, [mapping_tuple_label_to_int_concepts[concept_str] for concept_str in tuple_concepts]) 
                for label, tuple_concepts in map_int_label_to_tuple.items())
            # {0: [1, 10], ...}

            preprocess_concept_to_integer(train_img_info, mapping_tuple_label_to_int_concepts)
            preprocess_concept_to_integer(val_img_info, mapping_tuple_label_to_int_concepts)
            preprocess_concept_to_integer(test_img_info, mapping_tuple_label_to_int_concepts)

            preprocess_label_to_integer(train_img_info, map_tuple_label_to_int, prefix='continual/train/')
            preprocess_label_to_integer(val_img_info, map_tuple_label_to_int, prefix='continual/val/')
            preprocess_label_to_integer(test_img_info, map_tuple_label_to_int, prefix='continual/test/')

            '''if num_samples_each_label provided, sample images to balance each class for train set'''
            selected_train_images = []
            if num_samples_each_label is not None and num_samples_each_label > 0:
                imgs_each_label = dict()
                for item in train_img_info:
                    label = item['label']
                    if label in imgs_each_label:
                        imgs_each_label[label].append(item)
                    else:
                        imgs_each_label[label] = [item]
                build_in_seed = 1234
                build_in_rng = np.random.RandomState(seed=build_in_seed)
                for label, imgs in imgs_each_label.items():
                    selected_idxs = build_in_rng.choice(
                        np.arange(len(imgs)), num_samples_each_label,
                        replace=True if num_samples_each_label > len(imgs) else False)
                    for idx in selected_idxs:
                        selected_train_images.append(imgs[idx])
            else:
                selected_train_images = train_img_info

            '''generate train_list and test_list: list with img tuple (path, label)'''
            train_list = formulate_img_tuples(selected_train_images)
            val_list = formulate_img_tuples(val_img_info)
            test_list = formulate_img_tuples(test_img_info)
            # [('continual/val/59767.jpg', 0),...

            '''shuffle the train set'''
            if shuffle:
                rng = np.random.RandomState(seed=seed)
                order = np.arange(len(train_list))
                rng.shuffle(order)
                train_list = [train_list[idx] for idx in order]

            '''generate train_set and test_set using PathsDataset'''
            train_set = self.PathsDataset(
                root=img_folder_path,
                files=train_list,
                transform=transforms.Compose([transforms.Resize(image_size)]),
                loaded=load_set in ['train', 'all'],
                name='con_train',
            )
            val_set = self.PathsDataset(
                root=img_folder_path,
                files=val_list,
                transform=transforms.Compose([transforms.Resize(image_size)]),
                loaded=load_set in ['val', 'all'],
                name='con_val',
            )
            test_set = self.PathsDataset(
                root=img_folder_path,
                files=test_list,
                transform=transforms.Compose([transforms.Resize(image_size)]),
                loaded=load_set in ['test', 'all'],
                name='con_test',
            )

            datasets = {'train': train_set, 'val': val_set, 'test': test_set}
            meta_info = {
                "label_set": label_set, 
                "map_tuple_label_to_int": map_tuple_label_to_int, 
                "map_int_label_to_tuple": map_int_label_to_tuple, 
                "concept_set": concept_set,
                "mapping_tuple_label_to_int_concepts": mapping_tuple_label_to_int_concepts,
                "map_int_concepts_label_to_str": map_int_concepts_label_to_str,
                "map_int_label_to_concept": map_int_label_to_concept, 
                "train_list": train_list, "val_list": val_list, "test_list": test_list, 
                "img_folder_path": img_folder_path,}

        elif mode in ['sys', 'pro', 'non', 'noc']:  # no sub
            json_name = {'sys': 'O365_sys_fewshot_crop.json', 'pro': 'O365_pro_fewshot_crop.json',
                         'non': 'O365_non_fewshot_crop.json', 'noc': 'O365_noc_fewshot_crop.json'}[mode]
            json_path = os.path.join(img_folder_path, json_name)
            with open(json_path, 'r') as f:
                img_info = json.load(f)
            label_set = sorted(list(set([tuple(sorted(item['label'])) for item in img_info])))
            map_tuple_label_to_int = dict((item, idx + label_offset) for idx, item in enumerate(label_set))
            map_int_label_to_tuple = dict((idx + label_offset, item) for idx, item in enumerate(label_set))
            concept_set = sorted(list(set([concept for item in img_info for concept in (
                item['label'] if isinstance(item['label'], list) else [item['label']])])))
            logging.info(f'concept_set: {concept_set}')
            mapping_tuple_label_to_int_concepts = dict((item, idx) for idx, item in enumerate(concept_set))
            map_int_concepts_label_to_str = dict((idx, item) for idx, item in enumerate(concept_set))
            preprocess_concept_to_integer(img_info, mapping_tuple_label_to_int_concepts)
            preprocess_label_to_integer(img_info, map_tuple_label_to_int, prefix=f'fewshot/{mode}/')
            map_int_label_to_concept = dict(
                (label, [mapping_tuple_label_to_int_concepts[concept_str] for concept_str in tuple_concepts]) 
                for label, tuple_concepts in map_int_label_to_tuple.items())
            # {0: [1, 10], ...}
            img_list = formulate_img_tuples(img_info)
            dataset = self.PathsDataset(
                root=img_folder_path,
                files=img_list,
                transform=transforms.Compose([transforms.Resize(image_size)]),
                loaded=True,
                name=f'few_{mode}',
            )

            datasets = {'dataset': dataset}
            meta_info = {
                "label_set": label_set, 
                "map_tuple_label_to_int": map_tuple_label_to_int, 
                "map_int_label_to_tuple": map_int_label_to_tuple, 
                "concept_set": concept_set,
                "mapping_tuple_label_to_int_concepts": mapping_tuple_label_to_int_concepts,
                "map_int_concepts_label_to_str": map_int_concepts_label_to_str,
                "map_int_label_to_concept": map_int_label_to_concept,
                "img_list": img_list,
                "img_folder_path": img_folder_path,}

        else:
            raise Exception(f'Un-implemented mode "{mode}".')

        return datasets, meta_info