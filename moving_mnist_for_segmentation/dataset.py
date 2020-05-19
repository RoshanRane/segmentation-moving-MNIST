#!/usr/bin/python3

import math
import os
import sys
import numpy as np
from PIL import Image
from skimage.draw import random_shapes
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import gzip
if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

class Dataset:
    
    def __init__(self):
        # loads mnist from web on demand if not already downloaded
        if not os.path.exists("mnist"): os.mkdir("mnist")
            
        for filename in ["t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", 
                          "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"]:
            if not os.path.exists("mnist/"+filename):
                print("Downloading %s" % filename)
                urlretrieve("http://yann.lecun.com/exdb/mnist/"+filename, "mnist/"+filename)

    
    def _load_mnist(self, split): 
        
        if split == "test":
            with gzip.open('mnist/t10k-images-idx3-ubyte.gz', 'rb') as f:
                X = np.frombuffer(f.read(), np.uint8, offset=16)
            X = X.reshape(-1, 1, 28, 28) / np.float32(256)
            with gzip.open('mnist/t10k-labels-idx1-ubyte.gz', 'rb') as f:
                y = np.frombuffer(f.read(), np.uint8, offset=8)
        else:
            
            with gzip.open('mnist/train-images-idx3-ubyte.gz', 'rb') as f:
                X = np.frombuffer(f.read(), np.uint8, offset=16)
            X = X.reshape(-1, 1, 28, 28) / np.float32(256)
            with gzip.open('mnist/train-labels-idx1-ubyte.gz', 'rb') as f:
                y = np.frombuffer(f.read(), np.uint8, offset=8)
            
            if split == "train":
                X = X[:-5000]
                y = y[:-5000]
            elif split == "val":
                X = X[-5000:]
                y = y[-5000:]
            else:
                raise ValueError("invalid value provided for 'split'.\
Allowed values are 'train', 'val', or 'test'")

        return X, y
    
    # helper function to standardize images
    def _img_std(self, img, mean=0, std=1):
        '''
        Args:
            img: image array of shape C x W x H
            mean: Mean to add
            std: Standard Deviation to add
        Returns:
            Image with dimensions H x W x C or H x W if it's a single channel image
        '''
        img_std = (((img + mean)*255.)*std).astype(np.uint8)
        img_std = img_std.clip(0, 255).transpose(2,1,0).squeeze()
        return img_std
    
    def generate(
        self,
        split="train",
        frame_size=(64, 64),
        num_videos=100, 
        num_frames=30, 
        min_ndigits=2, max_ndigits=2,
        ignore_label=255,
#         rotatehint=False, # TODO
    #     mnist_size_randomize=False, # TODO 
    ):
        '''
        Args:
            split: One of ("train", "val", "test"). The split of the MNIST digits dataset to use for generating the video.
            frame_size: Shape of the frames of the video (frame_width, frame_height)
            num_videos: Number of videos of moving MNIST
            num_frames: Number of frames in each video
            min_ndigits, max_ndigits: 
                The number of digits moving around in the video is randomly selected between (min_ndigits, max_ndigits)
            ignore_label: The value to fill-in for dummy values in labels
        Returns:
            Dataset with shape (num_videos x num_frames x 1 x frame_width x frame_height) 
            Segmentation Masks with shape (num_videos x num_frames x 1 x frame_width x frame_height) 
            Digit labels with shape (num_videos x max_ndigits)
        '''
        mnist, mnist_lab = self._load_mnist(split.lower()) 


        assert len(frame_size) == 2
        height, width = frame_size
        n, _, mnist_height, mnist_width = mnist.shape
        assert min_ndigits>=1 and min_ndigits<=max_ndigits
        # Get how many pixels can we move around a single image
        lims = (x_lim, y_lim) = width - mnist_width, height - mnist_height

        # Create a dataset of shape of num_videos X num_frames x 1 x new_width x new_height Eg : 20000 x 20 x 1 x 64 x 64
        dataset = np.empty((num_videos, num_frames, 1, height, width), dtype=np.uint8)
        dataset_masks = np.empty((num_videos, num_frames, 1, height, width), dtype=np.uint8)
        dataset_labels = np.full((num_videos, num_frames,  max_ndigits), fill_value=ignore_label, dtype=np.uint8)

        for vid_idx in tqdm(range(num_videos)):
            # randomly select the number of nums per image
            nums_per_image = np.random.randint(min_ndigits, max_ndigits+1)
            # Randomly generate velocity (direction + speed), 
            direcs = np.pi * (np.random.rand(nums_per_image)*2 - 1)
            speeds = np.random.randint(5, size=nums_per_image) + 2
            veloc = np.asarray(
                [(speed*math.cos(direc), speed*math.sin(direc)) for direc, speed in zip(direcs, speeds)]
            )
            # Randomly generate brightness for all images
            rand_brightness = []
            for i in range(nums_per_image):
                start = np.random.uniform(0.1, 1.5)
                stop = np.random.uniform(0.1, 1.5) 
                step = (stop-start)/num_frames
                rand_brightness.append(np.arange(start, stop, step))

            # Get a list of MNIST images randomly sampled from the dataset
            rand_idx = np.random.randint(0, mnist.shape[0], nums_per_image)
            mnist_images = [Image.fromarray(self._img_std(mnist[s])) for s in rand_idx]
            mnist_labels = [mnist_lab[s] for s in rand_idx]

            # prepare a static noisy canvas for each video
            canvas_x_lims, canvas_y_lims = height//3,width//3
            static_canvas_distractors,_ = random_shapes((height+canvas_x_lims, width+canvas_y_lims), 
                                                        min_shapes=10, max_shapes=25, 
                                                        multichannel=False, allow_overlap=True) 

            # Generate initial positions for nums_per_image as tuples (x,y)
            positions = np.asarray(
                [(np.random.rand() * x_lim, np.random.rand() * y_lim) for _ in range(nums_per_image)]
            )
            # Generate initial rotation angle for nums_per_image
            rotates = np.random.randint(0, 360, size=nums_per_image)
#             if rotatehint:
#                 rotate_increments = np.array(mnist_labels, dtype=int)*2
#             else:
            rotate_increments = np.random.randint(0, 10, size=nums_per_image)
            canvas_position = np.random.randint(0, high=canvas_x_lims//2, size=(2,))
            # Generate new frames for the entire num_frames
            for frame_idx in range(num_frames): 
                canvas = static_canvas_distractors.copy()
                # move the bg canvas a little, if possible
                if canvas_position[0]+2 < canvas_x_lims:
                    canvas_position[0] = canvas_position[0] + np.random.randint(0,3)
                if canvas_position[1]+2 < canvas_y_lims:
                    canvas_position[1] = canvas_position[1] + np.random.randint(0,3)                
                canvas = canvas[
                    canvas_position[0]:canvas_position[0]+height, canvas_position[1]:canvas_position[1]+width
                ]

                canvas_mask = np.full((height, width), fill_value=ignore_label, dtype=np.float32)

                # Super impose images on the canvas
                for i in range(nums_per_image):
                    # generate random noise for the canvas background
                    noise = (np.random.normal(loc=127, scale=127, size=(height, width))).astype(np.uint8)
                    canvas[canvas==255] = noise[canvas==255]
                    canvas = canvas.astype(np.float32)
                    # copy the MNIST image
                    mnist_rotated = mnist_images[i].rotate(rotates[i].astype(int))
                    mnist_copy = Image.new('L', (height, width))
                    mnist_copy.paste(mnist_rotated, tuple(positions[i].astype(int)))
                    mnist_copy = np.array(mnist_copy)*rand_brightness[i][frame_idx]
                    canvas[mnist_copy>0]=mnist_copy[mnist_copy>0]

                    # if there are regions of overlap in canvas_mask, override with the last label
                    canvas_mask[mnist_copy>0] = mnist_labels[i]

                # Get the next position by adding velocity
                next_pos = positions + veloc

                rotates = (rotates + rotate_increments)%360
                # Iterate over velocity and see if we hit the wall. If yes, change direction
                for i, pos in enumerate(next_pos):
                    for j, coord in enumerate(pos):
                        if coord < -2 or coord > lims[j] + 2:
                            veloc[i] = list(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j + 1:]))
                # Make the permanent change to position by adding updated velocity
                positions = positions + veloc

                # Add the canvas to the dataset array
                dataset[vid_idx, frame_idx, 0] = canvas.clip(0, 255).astype(np.uint8) 
                dataset_masks[vid_idx, frame_idx, 0] = (canvas_mask).astype(np.uint8)  
            # Add the label to the dataset labels array
            dataset_labels[vid_idx, :nums_per_image] = mnist_labels
        # store for class
        self.cur_dataset = dataset
        self.cur_dataset_masks = dataset_masks
        self.cur_dataset_labels = dataset_labels
        self.cur_ignore_label = ignore_label
        self.cur_split = split.lower() 
        return dataset, dataset_masks, dataset_labels

    def display_samples(self, n=3):
        ''' Display 'n' random videos and the segmentation masks from the datasets '''
        cmap = "magma"
        num_videos, num_frames, _, _, _ = self.cur_dataset.shape
        num_frames = 10 if num_frames>10 else num_frames
        
        plot_vids = np.random.randint(0, num_videos, size=n)
        for i,plot_vid in enumerate(plot_vids):
            lbls = list(self.cur_dataset_labels[plot_vid,0])
            if 10 in lbls: lbls.remove(10)
            fig = plt.figure(figsize=(5*num_frames,10))
            plt.suptitle("Sample {} : MNIST digits {}".format(i+1, lbls), x=0.23, y=0.95, fontsize=40)
            fig.subplots_adjust(wspace = 0.01, hspace = 0.02)
            for i in range(num_frames):
                plt.subplot(2, num_frames, i+1).axis('off')
                plt.imshow(self.cur_dataset[plot_vid,i,0], cmap="gray")
                plt.subplot(2, num_frames, num_frames+i+1).axis('off')
                mask = (self.cur_dataset_masks[plot_vid,i][0,:,:]+3)
                mask[mask==(self.cur_ignore_label+3)]=0
                im = plt.imshow(mask, cmap=cmap, vmin=0, vmax=14)
        # plot color map    
        gradient = np.linspace(3, 14, 10)
        gradient = np.vstack((gradient, gradient))
        fig, ax = plt.subplots(figsize=(3.2,0.3))
        fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
        ax.set_title('colormap used to denote the MNIST digits\n0   1   2   3   4   5   6   7   8   9', fontsize=12, loc='left')
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(cmap), vmin=0, vmax=16)
        ax.set_axis_off()
        plt.margins(0,0)
        
        
    def save(self, path, filetype):
        '''Save the generated dataset and labels as raw npy arrays or as compressed .h5 files'''
        if not os.path.isdir(path):
            print("creating folder ", path)
            os.makedirs(path)
    
        # save images and labels
        if filetype == 'npy':
            np.save(path+self.cur_split+"_images", self.cur_dataset)
            np.save(path+self.cur_split+"_masks", self.cur_dataset_masks)
            np.save(path+self.cur_split+"_labels", self.cur_dataset_labels)
        elif filetype == 'h5':
            import h5py
            with h5py.File(path+self.cur_split+".h5", "w") as f:
                f.create_dataset("images", data=self.cur_dataset, compression="gzip")
                dataset_mask = f.create_dataset('masks', data=self.cur_dataset_masks, compression="gzip")
                dataset_labels = f.create_dataset('labels', data=self.cur_dataset_labels)         
        else:
            raise NotImplementedError("filetype not supported")
        