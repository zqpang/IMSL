from __future__ import absolute_import
import os.path as osp
from torch.utils.data import Dataset
from PIL import Image



def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img



def read_parsing_result(img_path):
    """Keep reading image until succeed.
        This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img




class Preprocessor(Dataset):
    def __init__(self, dataset, train = True, root=None, transform1=None, transform2=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform1 = transform1
        self.transform2 = transform2
        self.train = train
        
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = read_image(fpath)
        
        if self.train:
            fname_mask = fname.split('/')

            if 'rgbnt201' in fname_mask:
                mask_path = '/home/userroot/database2/zhiqi/dataset/rgbnt201/mask'
                fname_mask = osp.join(mask_path,fname_mask[-1][:-3] + 'png')
                img_mask = read_parsing_result(fname_mask)        
        
        
        if self.transform2 is not None:
            img1 = self.transform2(img)
        else:
            img1 = self.transform1(img)
        
        if self.train:
            img_mask = self.transform1(img_mask)
        
        else:
            img_mask = img1
        
        return img1, img_mask, fname, pid, camid, index
    


class Train_Preprocessor(Dataset):
    def __init__(self, dataset, train = True, root=None, transform1=None, transform2=None):
        super(Train_Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform1 = transform1
        self.transform2 = transform2
        self.train = train
        
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pseudo_label, camid, img_index, accum_label = self.dataset[index]
        
        
        
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = read_image(fpath)
        
        if self.train:
            fname_mask = fname.split('/')

            if 'rgbnt201' in fname_mask:
                mask_path = '/home/userroot/database2/zhiqi/dataset/rgbnt201/mask'
                fname_mask = osp.join(mask_path,fname_mask[-1][:-3] + 'png')
                img_mask = read_parsing_result(fname_mask)        
        
        if self.transform2 is not None:
            img1 = self.transform2(img)
        else:
            img1 = self.transform1(img)
        
        if self.train:
            img_mask = self.transform1(img_mask)
        
        else:
            img_mask = img1
        
        return img1, img_mask, fname, pseudo_label, camid, img_index, accum_label        
