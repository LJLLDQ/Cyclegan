import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        #print('sss',A_path)
        

        # with rawpy.imread(A_path) as raw:
        #     A_img = raw.postprocess(use_camera_wb=True,  use_auto_wb=False,exp_shift=3 )

        
        # with rawpy.imread(B_path) as raw:
        #     B_img = raw.postprocess(use_camera_wb=True,  use_auto_wb=False,exp_shift=3 )

        if 'ARW' in A_path:
            raw = rawpy.imread(A_path)
            # input_full = np.expand_dims(input(raw),axis=0) *ratio	
            A_img = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            # A_img = Image.open(A_img).convert('RGB')
            #print(A_img)
            #PIL_image = Image.fromarray(A_img)
            #print(tmp)
            A_img = Image.fromarray(np.uint8(A_img))
        
        else:
             A_img = Image.open(A_path).convert('RGB')

        if 'ARW' in B_path:
            raw = rawpy.imread(B_path)
            # input_full = np.expand_dims(input(raw),axis=0) *ratio	
            B_img = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            # B_img = Image.open(B_img).convert('RGB')
            B_img = Image.fromarray(np.uint8(B_img))

        else:
            B_img = Image.open(B_path).convert('RGB')

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        
        #PIL_image = Image.fromarray(B_img)
        #print(B_img.shape)



 
        # apply image transformation

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
