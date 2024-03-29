# fork of https://github.com/eriklindernoren/PyTorch-Deep-Dream

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os
import tqdm
import scipy.ndimage as nd
from torchvision import transforms
import sound_functions as sf
import torch_filter_classifier as classifier

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def deprocess(image_np):
    image_np = image_np.squeeze().transpose(1, 2, 0)
    image_np = image_np * std.reshape((1, 1, 3)) + mean.reshape((1, 1, 3))
    image_np = np.clip(image_np, 0.0, 255.0)
    return image_np


def clip(image_tensor):
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[0, c] = torch.clamp(image_tensor[0, c], -m / s, (1 - m) / s)
    return image_tensor


def dream(sound, model, iterations, lr):
    """ Updates the image to maximize outputs for n iterations """
    
    Tensor = torch.FloatTensor
    print('DREAMING')
    print('iterations', iterations)
    sound = Variable(Tensor(sound), requires_grad=True)

    for i in range(iterations):
        print('dream i', i)
        model.zero_grad()
 
        out = model(sound)

        _, predicted = torch.max(out.data, 1)
        print('predicted class', predicted)
        loss = out.norm()
        loss.backward()

        grad = sound.grad.data.cpu().numpy()
        # print(np.max(grad), np.min(grad), np.average(grad))
        # print('grad shape', grad.shape)
        # print('grad nonzero', np.count_nonzero(grad))
        # print('grad size', grad.size)

        #avg_grad = np.abs(sound.grad.data.cpu().numpy()).mean(axis=0)
        norm_lr = grad * lr
        print('grad max', np.max(grad))
        print('norm_lr max', np.max(norm_lr))
        s = sound.cpu().data.numpy()
        print('sound max', np.max(s))

        # print('norm_lr shape', norm_lr.shape)
        # print('norm_lr nonzero', np.count_nonzero(norm_lr))
        # print('norm_lr size', norm_lr.size)
        # exit()

        # print('avg_grad')
        # print(avg_grad)
        # print('norm_lr')
        # print(norm_lr)
        # exit()

        norm_lr = Tensor(norm_lr)
        sound.data += norm_lr * sound.grad.data
        # sound.data = clip(sound.data) #TMPDEBUG don't know what this does
        sound.grad.data.zero_()
    return sound.cpu().data.numpy()


def deep_dream(sound, model, iterations, lr, octave_scale, num_octaves):
    """ Main deep dream method """
    #image = preprocess(image).unsqueeze(0).cpu().data.numpy()

    # Extract image representations for each octave
    # octaves = [sound]


    # print('octaves - 1 shape', octaves[-1].shape)
    # for _ in range(num_octaves - 1):
    #     octaves.append(nd.zoom(octaves[-1], (1, 1, 1 / octave_scale), order=1))

    # detail = np.zeros_like(octaves[-1])

    # print('octaves shapes')
    # for oct in range(len(octaves)):
    #     print(oct)
    #     print (octaves[oct].shape)

    # for octave, octave_base in enumerate(tqdm.tqdm(octaves[::-1], desc="Dreaming")):
    #     if octave > 0:
    #         # Upsample detail to new octave dimension
    #         detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)
    #     # Add deep dream detail from previous octave to new base
    #     input_sound = octave_base + detail

    #     # Get new deep dream image
    #     dreamed_sound = dream(input_sound, model, iterations, lr)
    #     # Extract deep dream details
    #     detail = dreamed_sound - octave_base

    dreamed_sound = dream(sound, model, iterations, lr)

    print('shape', dreamed_sound.shape)

    return dreamed_sound


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, default="images/supermarket.jpg", help="path to input image")
    parser.add_argument("--iterations", default=20, help="number of gradient ascent steps per octave")
    parser.add_argument("--at_layer", default=27, type=int, help="layer at which we modify image to maximize outputs")
    parser.add_argument("--lr", default=1, help="learning rate")
    parser.add_argument("--octave_scale", default=1.4, help="image scale between octaves")
    parser.add_argument("--num_octaves", default=10, help="number of octaves")
    args = parser.parse_args()

    # # Load image
    # image = Image.open(args.input_image)
    # Load sound


    wav_data = sf.getWav('../sounds/songsinmyhead/b/02whichdescribeshowyourfeeling.wav')
    #wav_data = sf.getWav('../sounds/plum_island/plum_island.wav')
    
    wav_data = wav_data[44100:] # trim possible silence

    # # #RANDOM INPUT
    wav_data = np.random.random_sample((wav_data.shape))
    wav_data *= 2
    wav_data -= 1
    wav_data *= 0.01
    # print('wav data min max avg', np.min(wav_data), np.max(wav_data), np.average(wav_data))

    # # ZERO INPUT
    # wav_data = np.zeros(wav_data.shape)

    fft_data = sf.getFFT(wav_data, classifier.fftwidth, classifier.timewidth)
    sound = fft_data[:5]
    print('orig sound max', np.max(sound))
 
   # sound = sf.filterBank(sound, order=2, n=filter_n, step=filter_step)
    
    # batch
    # sound = sound.T
    # sound = sound.reshape(items, clip_size, filter_n)
    # sound = np.swapaxes(sound, 1, 2)
    # print('sound shape', sound.shape)

    # Define the model
    # network = models.vgg19(pretrained=True)
    
    model = classifier.loadNet() #TMPDEBUG does this work?
    # layers = list(network.children())
    # print(len(layers))
    # exit()
    # print(layers)
    # model = nn.Sequential(*layers[: (args.at_layer + 1)])
 

    print(model)
    dreamed_sound = sound #TMPDEBUG

    # Extract deep dream sound
    dreamed_sound = deep_dream(
        sound,
        model,
        iterations=int(args.iterations),
        lr=float(args.lr),
        octave_scale=float(args.octave_scale),
        num_octaves=int(args.num_octaves),
    )





    # Save sound
    os.makedirs("outputs", exist_ok=True)
    filename = 'outputs/dreamsound.wav'


    # undo batching
    # dreamed_sound = np.swapaxes(dreamed_sound, 1, 2)
    # dreamed_sound = dreamed_sound.reshape((-1, filter_n))
    # dreamed_sound = dreamed_sound.T
    # print('dreamed_sound shape', dreamed_sound.shape)


    # dreamed_sound = np.sum(dreamed_sound, axis=0)

    dreamed_sound = sf.toTimeDomain(dreamed_sound)

    dreamed_sound /= 0.001 + np.max(np.absolute(dreamed_sound))
    dreamed_sound *= 0.7
    print('dreamed_sound max, min, avg', np.max(dreamed_sound), np.min(dreamed_sound), np.average(dreamed_sound))
   
    sf.writeWav(dreamed_sound, filename)
    print('wrote ', filename)
