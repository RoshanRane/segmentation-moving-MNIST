---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
#!/usr/bin/python3
import math
import os
import sys
import numpy as np
from PIL import Image
from skimage.draw import random_shapes
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
%matplotlib inline
```

## Class definition

```python
# helper functions
def arr_to_pilarr(img, mean=0, std=1):
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


# loads mnist from web on demand
def load_dataset():
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
    
    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def generate_moving_mnist(
    split="train",
    shape=(64, 64), 
    num_frames=30, 
    num_videos=100, 
    ignore_label=255,
#     mnist_size_randomize=False, # TODO 
    min_nums_per_image=2, max_nums_per_image=2,
    rotatehint=False):
    '''
    Args:
        training: Boolean, used to decide if downloading/generating train set or test set
        shape: Shape we want for our moving images (new_width and new_height)
        num_frames: Number of frames in each video
        num_videos: Number of videos of moving MNIST
        original_size: Real size of the images (eg: MNIST is 28x28)
        nums_per_image: Digits per movement/animation/gif. 
        If a list of possible nums_per_image is provided, the number is sampled randomly from the list per video
    Returns:
        Dataset of np.uint8 type with dimensions num_videos x num_frames x 1 x new_width x new_height
    '''
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset() 
    
    if "train" in split:
        mnist = X_train
        mnist_lab = y_train
    elif "val" in split:
        mnist = X_val
        mnist_lab = y_val
    elif "test" in split:
        mnist = X_test
        mnist_lab = y_test
    else:
        raise ValueError("allowed values for 'split' are ['train','test','val'] but given {}".format(split))
    
    height, width = shape
    n, _, mnist_height, mnist_width = X_test.shape
    assert min_nums_per_image>=1 and min_nums_per_image<=max_nums_per_image
    # Get how many pixels can we move around a single image
    lims = (x_lim, y_lim) = width - mnist_width, height - mnist_height
    
    # Create a dataset of shape of num_videos X num_frames x 1 x new_width x new_height Eg : 20000 x 20 x 1 x 64 x 64
    dataset = np.empty((num_videos, num_frames, 1, height, width), dtype=np.uint8)
    dataset_masks = np.empty((num_videos, num_frames, 1, height, width), dtype=np.uint8)
    dataset_labels = np.full((num_videos, num_frames,  max_nums_per_image), fill_value=ignore_label, dtype=np.uint8)

    for vid_idx in tqdm(range(num_videos)):
        # randomly select the number of nums per image
        nums_per_image = np.random.randint(min_nums_per_image, max_nums_per_image+1)
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
        mnist_images = [Image.fromarray(arr_to_pilarr(mnist[s])) for s in rand_idx]
        mnist_labels = [mnist_lab[s] for s in rand_idx]
        
        # prepare a static noisy canvas for each video
        canvas_x_lims, canvas_y_lims = 20,20
        static_canvas_distractors,_ = random_shapes((height+canvas_x_lims, width+canvas_y_lims), 
                                                    min_shapes=15, max_shapes=25, 
                                                    multichannel=False, allow_overlap=True) 
        
        # Generate initial positions for nums_per_image as tuples (x,y)
        positions = np.asarray(
            [(np.random.rand() * x_lim, np.random.rand() * y_lim) for _ in range(nums_per_image)]
        )
        # Generate initial rotation angle for nums_per_image
        rotates = np.random.randint(0, 360, size=nums_per_image)
        if rotatehint:
            rotate_increments = np.array(mnist_labels, dtype=int)*2
        else:
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
            dataset_labels[vid_idx, frame_idx, :nums_per_image] = mnist_labels
            
    return dataset, dataset_masks, dataset_labels
```

## Configuration

```python
frame_size= 64
num_frames= 20  # length of each sequence
mnist_size_randomize = False  # to randomize the size of mnist digit within frame
min_nums_per_image = 2  # number of digits in each frame
max_nums_per_image = 3  # number of digits in each frame
ignore_label = 10
split = "val" 
num_videos = 2500
path = "./data/"
filetype = "h5"
```

```python
dat, dat_masks, dat_labels = generate_moving_mnist(split, shape=(frame_size, frame_size),
                                        ignore_label=ignore_label,
                                        num_frames=num_frames, num_videos=num_videos,
                                        min_nums_per_image=min_nums_per_image, max_nums_per_image=max_nums_per_image,
                                        rotatehint=False)

print(dat.shape, dat_masks.shape, dat_labels.shape)
```

## Visualization of some samples

```python
cmap = "magma"

num_frames= 10 if num_frames>10 else num_frames
for i in range(3):
    plot_vid = np.random.randint(0, num_videos)
    lbls = list(dat_labels[plot_vid,0])
    if 10 in lbls: lbls.remove(10)
    fig = plt.figure(figsize=(5*num_frames,10))
    plt.suptitle("Sample {} : MNIST digits {}".format(i+1, lbls), 
                                                x=0.23, y=0.95, fontsize=40)
    fig.subplots_adjust(wspace = 0.01, hspace = 0.02)
    for i in range(num_frames):
        plt.subplot(2, num_frames, i+1).axis('off')
        plt.imshow(dat[plot_vid,i,0], cmap="gray")
        plt.subplot(2, num_frames, num_frames+i+1).axis('off')
        mask = (dat_masks[plot_vid,i][0,:,:]+3)
        mask[mask==(ignore_label+3)]=0
        plt.imshow(mask, cmap=cmap, vmin=0, vmax=14)
plt.margins(0,0)

# plot color map    
gradient = np.linspace(3, 14, 10)
gradient = np.vstack((gradient, gradient))
fig, ax = plt.subplots(figsize=(3.2,0.3))
fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
ax.set_title('colormap used to denote the MNIST digits\n0   1   2   3   4   5   6   7   8   9', fontsize=12, loc='left')
ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(cmap), vmin=0, vmax=16)
ax.set_axis_off()
plt.margins(0,0)
```

## Save 

```python
if not os.path.isdir(path):
    print("creating folder ", path)
    os.makedirs(path)
    
# save images and labels
if filetype == 'npy':
    np.save(path+split+"_images", dat)
    np.save(path+split+"_masks", dat_masks)
    np.save(path+split+"_labels", dat_labels)
elif filetype == 'h5':
    import h5py
    with h5py.File(path+split+".h5", "w") as f:
        f.create_dataset("images", data=dat)#, compression="gzip")
        dataset_mask = f.create_dataset('masks', data=dat_masks)#, compression="gzip")
        dataset_labels = f.create_dataset('labels', data=dat_labels)         
else:
    raise NotImplementedError("filetype not supported")
```
