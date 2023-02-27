#from pathlib import Path

import os

import numpy as np
import math
import random
import tqdm
import gc
import argparse
import time

#import matplotlib.pyplot as plt

import torch
#import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import transforms
from safetensors.torch import save_file

from diffusers.optimization import get_constant_schedule

#from diffusers import AutoencoderKL
import lpips

from transformers.optimization import Adafactor, AdafactorSchedule
#from sd_vae import AutoencoderKL

from PIL import Image
import cv2
#vaeロード面倒だからsd-scriptからそのまま借りる
from library.model_util import load_vae
import yaml

try:
    import torch.utils.tensorboard.writer as tensorboardX
except:
    pass
import datetime

def load_vae_from_sd_checkpoint(model_path, dtype=torch.float16, use_xformers: bool = True):
    print(f"load dtype: {dtype}")
    vae = load_vae(model_path, dtype=dtype)

    #xformersのフラグをオンにする
    for i in range(len(vae.encoder.mid_block.attentions)):
        vae.encoder.mid_block.attentions[i].set_use_memory_efficient_attention_xformers(use_xformers)
    for i in range(len(vae.decoder.mid_block.attentions)):
        vae.decoder.mid_block.attentions[i].set_use_memory_efficient_attention_xformers(use_xformers)
    
    print(f"xformers: {vae.decoder.mid_block.attentions[0]._use_memory_efficient_attention_xformers}")
    return vae

#from src.utils import ImageFolder, VAEHandler, denormalize, preprocess_images
class IMAGE_DIC():
    def __init__(self, file_path: str) -> None:
        self.size : tuple[int, int] = None
        self.org_size : tuple[int, int] = None
        self.path : str = file_path
        self.data = None
        self.latent = None
        self.ratio : float = None
        self.area_size : int = None
        self.ratio_error : float = None
        self.scale : float = None

class VAE_TRAIN_DATASET(Dataset):
    def __init__(self, data_path, batch_size=1, gradient_accumulation_steps=1, shuffle=True,
                 resolution=(256,256), min_resolution=(128,128), max_size=512, min_size=128, divisible=64, bucket_serch_step=1, make_clipping=0., make_clip_num=1) -> None:
        '''
        data_path = ディレクトリパス　か　画像のパスリスト
        '''
        super().__init__()
        #ファイルパスとか
        self.dataset_dir_path = data_path
        self.file_paths = None
        self.data_list = {}
        self.dots = ["png", "jpg"]
        #学習に関するデータセット設定
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._data_len : int = 0
        self._data_len_add : int = 0
        self.shuffle : bool = shuffle
        self.make_clipping = make_clipping
        if self.make_clipping >= 1.: self.make_clipping=0.
        self.make_clip_num = make_clip_num
        #画像サイズに関する変数
        self.resolution = resolution
        self.min_resolution = min_resolution
        self.max_area_size = (self.resolution[0]//divisible) * (self.resolution[1]//divisible)
        self.min_area_size = (self.min_resolution[0]//divisible) * (self.min_resolution[1]//divisible)
        self.max_size = max_size
        self.min_size = min_size
        self.divisible = divisible
        self.bucket_serch_step = bucket_serch_step
        #bucket関連の変数（画像読み込み時）
        self.buckets_lists = []
        self.area_size_list = []
        self.bucket_area_size_resos_list = []
        self.bucket_area_size_ratio_list = []
        self.add_index = []
        # bucket関連の変数（学習時に使うやつ）
        #index -> vsize key
        self.index_to_enable_bucket_list : list[tuple[int,int]] = []
        # vsize key ->  dataset key
        self.enable_bucket_vsize_to_resos_lens : dict[tuple[int,int], int] = {}
        self.enable_bucket_vsize_to_keys_list : dict[tuple[int,int], list] = {}
        # vsize key -> keys_list index (-> dataset)
        self.enable_bucket_vsize_to_keys_indexs : dict[tuple[int,int], list] = {}
        #画像データセット
        self.data_list : dict[str, IMAGE_DIC] = {}
        #
        self.image_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        
        #ファイルのパスをとりあえず取得する
        self.get_files_path()
        #先にデータリストの入れ物を作る
        self.make_datalist()
        #バケットを作成する
        self.make_buckets()
        #画像を読み込む
        self.load_images()
        #有効なバケット情報をまとめる
        self.create_enable_buckets()

        #テスト出力
        #for k, v in self.data_list.items():
        #    print(f"{k}: {v.org_size} -> {v.size} er({v.ratio_error})")
        #
    #読み込むデータリスト作成関連
    def get_files_path(self):
        file_paths = []
        if type(self.dataset_dir_path)==str:
            for root, dirs, files in os.walk(self.dataset_dir_path, followlinks=True):
                # ファイルを格納
                for file in files:
                    for dot in self.dots:
                        if dot in os.path.splitext(file)[-1]:
                            file_paths.append(os.path.join(root, file))
            self.file_paths = file_paths
        else:
            self.file_paths = self.dataset_dir_path
            self.dataset_dir_path = None
    def make_datalist(self):
        for file_path in self.file_paths:
            key = os.path.splitext(file_path)[0]
            if self.dataset_dir_path==None:
                key = os.path.basename(key)
            else:
                key = key[len(self.dataset_dir_path)+1:]
            img_data = IMAGE_DIC(file_path)
            self.data_list[key] = img_data
    # バケット作成
    def make_buckets(self):
        _max_area = self.max_area_size
        while _max_area >= self.min_area_size:
            resos = set()
            size = int(math.sqrt(_max_area)) * self.divisible
            resos.add((size, size))
            size = self.min_size
            while size <= self.max_size:
                width = size
                height = min(self.max_size, (_max_area // (size // self.divisible))*self.divisible)
                if height >= self.min_size:
                    resos.add((width, height))
                    resos.add((height, width))
                size += self.divisible
            resos = list(resos)
            resos.sort()

            self.area_size_list.append(_max_area)
            self.bucket_area_size_resos_list.append(resos)
            ratio = [w/h for w, h in resos]
            self.bucket_area_size_ratio_list.append(np.array(ratio))
            _max_area -= 1

        self.area_size_list = np.array(self.area_size_list)
    # 画像読み込み処理
    def load_image(self, img_path):
        image = Image.open(img_path)
        if not image.mode == "RGB":
            image=image.convert("RGB")
        return np.array(image, np.uint8)
    def load_images(self):
        print("画像読み込み中...")
        append_list = {}
        for key, img_data in tqdm.tqdm(self.data_list.items()):
            image_org = self.load_image(img_data.path)
            if not type(image_org)==np.ndarray:continue
            #画像サイズを決める
            img_data.org_size = [image_org.shape[1], image_org.shape[0]]
            img_data.size, img_data.ratio, img_data.ratio_error = self.sel_bucket_size(img_data.org_size[0], img_data.org_size[1])
            #画像加工処理
            image, img_data.scale = self.resize_image(image_org, img_data.size, img_data.ratio)
            image = self.image_transforms(image)
            img_data.data = image
            ############################################
            #クリッピング画像追加処理
            if self.make_clipping>0.:
                if img_data.scale<=self.make_clipping:
                    #脳筋処理方法(処理工数に無駄はあるけど確実　極端なクリッピングを防ぐためスケーリング処理が必要だった)
                    new_img = image_org
                    #クリッピングサイズが極端すぎないか確認する
                    if img_data.scale < 0.33:
                        new_scale = img_data.scale * 2 #おおよそ半解像度位をクリッピングするのが良さそうな気がする
                        resize = []
                        for i in range(2):
                            resize.append(int(new_img.shape[1-i] * new_scale + .5))
                        new_img = cv2.resize(new_img, resize, interpolation=cv2.INTER_AREA)
                        #print(f"resize: {img_data.scale} -> {new_scale}")
                    #追加リスト作成処理
                    for _ in range(self.make_clip_num):
                        i = 0
                        while True:
                            new_key = f"{key}+{i}"
                            if (not new_key in self.data_list) and (not new_key in append_list): break
                            i+=1
                        append_list[new_key] = IMAGE_DIC(new_key)
                        append_list[new_key].data = new_img
                        pos = []
                        for i in range(2):
                            pos.append(random.randint(0, append_list[new_key].data.shape[i]-img_data.size[1-i]-1))
                        append_list[new_key].data = append_list[new_key].data[pos[0]:pos[0]+img_data.size[1],pos[1]:pos[1]+img_data.size[0]]
                        #ここから下は情報を再取得していく
                        append_list[new_key].org_size = [append_list[new_key].data.shape[1], append_list[new_key].data.shape[0]]
                        append_list[new_key].size, append_list[new_key].ratio, append_list[new_key].ratio_error = self.sel_bucket_size(append_list[new_key].org_size[0], append_list[new_key].org_size[1])
                        append_list[new_key].data, append_list[new_key].scale = self.resize_image(append_list[new_key].data, append_list[new_key].size, append_list[new_key].ratio)
                        append_list[new_key].data = self.image_transforms(append_list[new_key].data)
        for k, v in append_list.items():
            self.data_list[k] = v

    def resize_image(self, image, _resized_size, ratio):
        img_size = image.shape[0:2]
        resized_size = [_resized_size[1], _resized_size[0]] #img_sizeからそのまま呼び出すと並びが　h,w なので処理をすっきりするために入れ替えておく
        #5/1=2.5 10/3=0.3 1/5=0.2 3/10=0.3
        re_retio = _resized_size[0] / _resized_size[1]
        if re_retio >= ratio:
            base_size = 1
        else:
            base_size = 0
        #cv2の方がPILより高品質な拡大縮小ができる
        resize_scale = resized_size[base_size] / img_size[base_size]
        resize = []
        for i in range(2):
            resize.append(int(img_size[1-i] * resize_scale + .5))
        if img_size[base_size] > resized_size[base_size]:
            image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)       # INTER_AREAでやりたいのでcv2でリサイズ
        elif img_size[base_size] < resized_size[base_size]:
            image = cv2.resize(image, resize, interpolation=cv2.INTER_CUBIC)#遅い代わりに高品質らしい
        #
        img_size = image.shape[0:2]
        p = [0, 0]
        for i in range(len(img_size)):
            if img_size[i] > resized_size[i]:
                trim_size = img_size[i] - resized_size[i]
                p[i] = trim_size // 2
        image = image[p[0]:p[0] + resized_size[0], p[1]:p[1] + resized_size[1]]
        assert image.shape[0] == resized_size[0] and image.shape[1] == resized_size[1], f"resized error {image.shape} to {resized_size}"
        return image, resize_scale

    def sel_bucket_size(self, img_width, img_height):
        area_size = (img_width//self.divisible) * (img_height//self.divisible)
        img_ratio = img_width / img_height
        area_size_er = self.area_size_list - area_size
        area_size_id = np.abs(area_size_er).argmin()
        area_size_id_list = [area_size_id]
        #探査範囲のsize id listを作成する
        for i in range(self.bucket_serch_step):
            if area_size_id -i <= 0:
                area_size_id_list.append(area_size_id+i+1)
            elif area_size_id + i + 1 >= len(self.bucket_area_size_resos_list):
                area_size_id_list.append(area_size_id-i-1)
            else:
                area_size_id_list.append(area_size_id-i-1)
                area_size_id_list.append(area_size_id+i+1)
        min_error = 10000
        min_area_size_id = area_size_id
        for area_size_id in area_size_id_list:
            area_ratio = self.bucket_area_size_ratio_list[area_size_id]
            ratio_errors = area_ratio - img_ratio
            ratio_error = np.abs(ratio_errors).min()
            if min_error > ratio_error:
                min_error = ratio_error
                min_area_size_id = area_size_id
            if min_error==0.:
                break
        area_size_id = min_area_size_id
        #ここから普通のバケットサイズ取得
        area_resos = self.bucket_area_size_resos_list[area_size_id]
        area_ratio = self.bucket_area_size_ratio_list[area_size_id]
        ratio_errors = area_ratio - img_ratio
        bucket_id = np.abs(ratio_errors).argmin()
        bucket_size = area_resos[bucket_id]

        return bucket_size, img_ratio, np.abs(ratio_errors).min()
    
    def make_latent(self, vae):
        print("latent作成中...")
        for img_data in tqdm.tqdm(self.data_list.values()):
            image = img_data.data
            image_tensor = image.unsqueeze(0).to(device=vae.device, dtype=vae.dtype)
            try:
                with torch.no_grad():
                    img_data.latent = vae.encode(image_tensor).latent_dist.mode().squeeze(0).to("cpu")
            except:
                print(f"error: {img_data.path} {image_tensor.size()}")
    #学習時に使うリストに関する関数
    def create_enable_buckets(self):
        for k, v in self.data_list.items():
            if not v.size in self.enable_bucket_vsize_to_keys_list:
                self.enable_bucket_vsize_to_keys_list[v.size] = [k]
            else:
                self.enable_bucket_vsize_to_keys_list[v.size].append(k)
        for k, v in self.enable_bucket_vsize_to_keys_list.items():
            count = len(v)
            self.enable_bucket_vsize_to_resos_lens[k] = count
            self.reset_indexs_list(k) #enable bucketsを作成する時に初期化しておく
            self._data_len += (count//self.batch_size) + (count%self.batch_size>0)
            for _ in range((count//self.batch_size) + (count%self.batch_size>0)):
                self.index_to_enable_bucket_list.append(k)
        #gradient accumulation stepsのための計算
        self._data_len_add = self.gradient_accumulation_steps - (self._data_len % self.gradient_accumulation_steps)

    def reset_indexs_list(self, vsize):
        now_list = [i for i in range(self.enable_bucket_vsize_to_resos_lens[vsize])]
        self.enable_bucket_vsize_to_keys_indexs[vsize]= now_list

        self.shuffle_indexs_list(vsize) #リセット時についでにシャッフルしたほうが楽
    def shuffle_indexs_list(self, vsize):
        if self.shuffle:
            now_list = self.enable_bucket_vsize_to_keys_indexs[vsize]
            random.shuffle(now_list)
            self.enable_bucket_vsize_to_keys_indexs[vsize] = now_list
        else: pass
    def reset_add_indexs_list(self):
        self.add_index = random.sample(range(self._data_len), self._data_len)
    #各バケット内の要素を取り出すためのkeyをindexとして扱うための関数
    def get_key(self, vsize):
        keys = self.enable_bucket_vsize_to_keys_list[vsize]
        key_index = self.enable_bucket_vsize_to_keys_indexs[vsize].pop(0) #popで取り出すので取り出した要素は消える
        #listの中を使い切ったら初期化して補充
        if len(self.enable_bucket_vsize_to_keys_indexs[vsize])==0:
            self.reset_indexs_list(vsize)
        return keys[key_index]
    
    def get_index_to_bucket_key(self, index):
        return self.index_to_enable_bucket_list[index]

    def __len__(self):
        return self._data_len + self._data_len_add
    def __getitem__(self, index):
        #まずはindexから使用するバケットサイズを決定する
        if index >= self._data_len:
            if len(self.add_index) == 0:
                self.reset_add_indexs_list()
            index = self.add_index.pop(0)
        vsize = self.get_index_to_bucket_key(index)

        teacher = []
        latents = []
        for i in range(self.batch_size):
            #取り出したvsize_keyから画像データにアクセスするためのkeyを取り出す
            key = self.get_key(vsize)
            img_data = self.data_list[key]
            latents.append(img_data.latent)
            teacher.append(img_data.data)
        latents = torch.stack(latents)
        teacher = torch.stack(teacher)

        data = {"latents": latents, "teacher": teacher}

        #動作確認用出力
        #print("=====================")
        #print(f"[{index}]: {key}")
        #print(self.data_list[key].data.size())
        #print(f"min: {self.data_list[key].data.min()} max: {self.data_list[key].data.max()}")

        return data

def save_vae(theta, output_file, save_type="safetensors"):
    vae_conversion_map = [
        ("conv_shortcut", "nin_shortcut"),
        ("conv_norm_out", "norm_out"),
        ("mid_block.attentions.0.", "mid.attn_1."),
    ]

    for i in range(4):
        # down_blocks have two resnets
        for j in range(2):
            hf_down_prefix = f"encoder.down_blocks.{i}.resnets.{j}."
            sd_down_prefix = f"encoder.down.{i}.block.{j}."
            vae_conversion_map.append((hf_down_prefix, sd_down_prefix))

        if i < 3:
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0."
            sd_downsample_prefix = f"down.{i}.downsample."
            vae_conversion_map.append((hf_downsample_prefix, sd_downsample_prefix))

            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix= f"up.{3-i}.upsample."
            vae_conversion_map.append((hf_upsample_prefix, sd_upsample_prefix))

        # up_blocks have three resnets
        # also, up blocks in hf are numbered in reverse from sd
        for j in range(3):
            hf_up_prefix = f"decoder.up_blocks.{i}.resnets.{j}."
            sd_up_prefix = f"decoder.up.{3-i}.block.{j}."
            vae_conversion_map.append((hf_up_prefix, sd_up_prefix))

    # this part accounts for mid blocks in both the encoder and the decoder
    for i in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{i}."
        sd_mid_res_prefix = f"mid.block_{i+1}."
        vae_conversion_map.append((hf_mid_res_prefix, sd_mid_res_prefix))

    vae_conversion_map_attn = [
        # (HF Diffusers, stable-diffusion)
        ("group_norm.", "norm."),
        ("query.", "q."),
        ("key.", "k."),
        ("value.", "v."),
        ("proj_attn.", "proj_out."),
    ]

    mapping = {k: k for k in theta.keys()}
    for k, v in mapping.items():
        for hf_part, sd_part  in vae_conversion_map:
            v = v.replace(hf_part, sd_part)
            mapping[k] = v
    for k, v in mapping.items():
        if "mid_block.attentions" in k:
            for hf_part, sd_part  in vae_conversion_map_attn:
                v = v.replace(hf_part, sd_part)
            mapping[k] = v
    new_state_dict = {v: theta[k] for k, v in mapping.items()}

    weights_to_convert = ["q", "k", "v", "proj_out"]
    for k, v in new_state_dict.items():
        for weight_name in weights_to_convert:
            if f"mid.attn_1.{weight_name}.weight" in k:
                new_state_dict[k] = new_state_dict[k].unsqueeze(dim=2).unsqueeze(dim=2)

    if save_type=="pt":
        torch.save({
                "state_dict": new_state_dict
                    }, output_file)
    else:
        save_file(new_state_dict, output_file)
    
    print(f"saved ... {output_file}")
    return

def collate_fn(examples):
  return examples[0]

#compVisが採用してるDiscriminatorそのまま
import functools
class NLayerDiscriminator(torch.nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, ndf=64, n_layers=3):
        super(NLayerDiscriminator, self).__init__()
        input_nc = 3
        norm_layer = torch.nn.BatchNorm2d
        use_bias = False
        kw = 4
        padw = 1
        sequence = [torch.nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), torch.nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                torch.nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                torch.nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            torch.nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            torch.nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            torch.nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = torch.nn.Sequential(*sequence)
    def forward(self, input):
        """Standard forward."""
        return self.main(input)
    def loss_func_discriminator(self, inputs, teacher):
        dis_fake = self(inputs)
        dis_true = self(teacher)
        loss_fake = torch.mean(torch.nn.functional.relu(1. + dis_fake)) * 0.5
        loss_real = torch.mean(torch.nn.functional.relu(1. - dis_true)) * 0.5
        return loss_fake, loss_real
    def loss_func(self, inputs, teacher):
        dis_fake = self(inputs)
        loss_real = -torch.mean(dis_fake)
        return loss_real

class Discriminator_Block(torch.nn.Module):
    def __init__(self, in_dims, hid_dims=16, down_scale=2) -> None:
        super(Discriminator_Block, self).__init__()
        self.conv_in = torch.nn.Conv2d(in_dims, hid_dims, 3, 1, 1)
        self.down1 = torch.nn.Conv2d(hid_dims, hid_dims, 3, down_scale)
        self.down2 = torch.nn.Conv2d(in_dims, hid_dims, 3, down_scale)
        self.activate1 = lambda x: x * torch.sigmoid(x)
        self.activate2 = lambda x: x * torch.sigmoid(x)
    def forward(self, inputs):
        h = self.conv_in(inputs)
        h = self.activate1(h)
        h = self.down1(h)
        h = self.activate2(h)
        res = self.down2(inputs)
        return (h + res) / math.sqrt(2)
        
class Discriminator(torch.nn.Module):
    def __init__(self, hid_dims=16, down_scale=2) -> None:
        super(Discriminator, self).__init__()
        self.conv1 = Discriminator_Block(3, hid_dims*2, down_scale)
        self.conv2 = Discriminator_Block(hid_dims*2, hid_dims, 1)
        self.conv_out = torch.nn.Conv2d(hid_dims, hid_dims, 1)
    def forward(self, inputs):
        h = self.conv1(inputs)
        h = self.conv2(h)
        h = self.conv_out(h)
        return h
    def loss_func_discriminator(self, inputs, teacher):
        dis_fake = self(inputs)
        dis_real = self(teacher)
        #loss_fake = F.softplus(dis_fake).mean()
        #loss_real = F.softplus(-dis_real).mean()
        loss_fake = torch.nn.functional.mse_loss(dis_fake, torch.zeros_like(dis_fake))
        loss_real = torch.nn.functional.mse_loss(dis_real, torch.ones_like(dis_real))
        return loss_fake, loss_real
    def loss_func(self, inputs, teacher):
        dis_fake = self(inputs)
        loss_real = torch.nn.functional.mse_loss(dis_fake, torch.ones_like(dis_fake))
        return loss_real

### cahiner時代にシンプルさを追及して作ったDiscriminator 精度は知らん
### conv2を積層にしたら広い範囲を見れる構造 ただその場合padding計算してskip connect必要
class SimpleDiscriminator(torch.nn.Module):
    def __init__(self, hid_dims=16, alpha=1.75) -> None:
        super(SimpleDiscriminator, self).__init__()
        self.alpha = alpha
        self.conv1 = torch.nn.Conv2d(3, hid_dims, 3, bias=False)
        self.activation1 = lambda x: x * torch.sigmoid(x)
        self.conv2 = torch.nn.Conv2d(hid_dims, hid_dims*2, 3, 1, 0, 2, bias=False)
        self.activation2 = lambda x: x * torch.sigmoid(x)
        self.conv3 = torch.nn.Conv2d(hid_dims*2, hid_dims, 3, bias=False)
    def forward(self, inputs):
        h = self.conv1(inputs)
        h = self.activation1(h)
        h = self.conv2(h)
        h = self.activation2(h)
        h = self.conv3(h)
        return h
    def loss_func_discriminator(self, inputs, teacher):
        dis_fake = self(inputs*self.alpha-teacher)
        dis_real = self(teacher*self.alpha-teacher)
        loss_fake = torch.nn.functional.softplus(dis_fake).mean()
        loss_real = torch.nn.functional.softplus(-dis_real).mean()
        #loss_real = F.softplus(-dis_real).mean()
        return loss_fake, loss_real
    def loss_func(self, inputs, teacher):
        dis_fake = self(inputs*self.alpha-teacher)
        loss_fake = torch.nn.functional.softplus(-dis_fake).mean()
        #loss_real = F.softplus(-dis_real).mean()
        return loss_fake

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
class print_command():
    DEL = "\033[2K\033[G"
def args_str_to_list(args_str):    
    tmp = tuple([int(r) for r in args_str.split(',')])
    if len(tmp) == 1:
        tmp = (tmp[0], tmp[0])
    return tmp
def gen_img(vae, latents, output_name):
    with torch.no_grad():
        test_img = vae.decode(latents).sample

    test_img = (test_img / 2 + 0.5).clamp(0, 1)
    test_img = test_img.cpu().permute(0, 2, 3, 1).float().numpy()
    test_img = (test_img * 255).round().astype("uint8")
    test_img = [Image.fromarray(im) for im in test_img]
    test_img[0].save(f"{output_name}.png")

def train(args):
    #必要な場合最初にseed値は固定してしまう
    if args.seed is not None:
        torch_fix_seed(args.seed)
    data_dir = args.dataset
    model_file = args.model
    batch_size = args.batch
    shuffle_flag = args.not_shuffle
    resolution = args_str_to_list(args.resolution)
    min_resolution = args_str_to_list(args.min_resolution)
    max_size = args.max_size
    min_size = args.min_size
    divisible = args.divisible
    bucket_serch_step = args.bucket_serch_step
    val_rate = args.val_rate
    make_clipping = args.make_clipping
    make_clip_num = args.make_clip_num

    max_grad_norm = args.grad_clip
    max_grad_norm_flag = (max_grad_norm>0.)
    max_data_loader_n_workers = args.max_data_loader_n_workers
    persistent_data_loader_workers = True
    n_workers = min(max_data_loader_n_workers, os.cpu_count() - 1)      # cpu_count-1 ただし最大で指定された数まで

    output_file = f"{args.output_file}.{args.save_type}"
    output_dir = os.path.split(output_file)[0]
    if not output_dir=="":
        if not os.path.isdir(output_dir):
            print(f"{output_dir} ディレクトリを作成しました")
            os.makedirs(output_dir)
    save_every_n_epoch = args.save_every_n_epoch
    pre_epoch = args.pre_epoch
    epoch = args.epoch
    gradient_accumulation_steps = args.gradient_accumulation_steps
    optimizer_type = args.optimizer
    optimizer_arg = {"lr": args.lr}
    warmup_init = args.warmup_init
    decoder_param_split = args.decoder_param_split

    latent_dropout = args.latent_dropout
    max_latent_dropout_epoch = args.max_latent_dropout_epoch
    if max_latent_dropout_epoch<=0:
        max_latent_dropout_epoch = epoch

    latent_noise_rate = args.latent_noise_rate

    rec_alpha_def = args.decode_alpha
    rec_alpha = args.decode_alpha
    rec_alpha_max = args.decode_alpha_max_epoch
    rec_alpha_linear = args.decode_alpha_linear
    rec_alpha_blight = 1.0 * (args.decode_alpha_blight)
    rec_alpha_step = 0.

    rec_rgb_alpha_def = []
    rec_rgb_alpha_step = []
    rec_rgb_alpha_max_epoch = args.decode_rgb_alpha_max_epoch
    rec_rgb_alpha_linear = args.decode_rgb_alpha_linear
    rec_rgb_alpha_blight = 1.0 * (args.decode_rgb_alpha_blight)
    rec_rgb_name = ["r", "g", "b"]
    
    l2_cost = args.l2_cost
    lpips_cost = args.lpips_cost
    disc_cost = args.discriminator_cost

    discriminator_type = args.discriminator_type
    discriminator_dim = args.discriminator_dim
    
    lpips_name = args.lpips_name

    enable_tensorboard = args.enable_tensorboard
    _t = datetime.datetime.today().strftime('%Y%m%d_%H%M')
    tensorboard_output = f"runs/vae_{_t}"
    del _t

    dtype = torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")             # "mps"を考量してない
    use_xformers_flag = args.not_use_xformers

    #######################################################################################
    if val_rate>0:
        pass
    #######################################################################################
    dataset = VAE_TRAIN_DATASET(data_dir, batch_size, gradient_accumulation_steps, shuffle_flag, resolution, min_resolution, max_size, min_size, divisible, bucket_serch_step, make_clipping, make_clip_num)

    if rec_alpha_max>0:
        rec_alpha_step = (rec_alpha-1.) / (rec_alpha_max * (len(dataset)/gradient_accumulation_steps))
    if rec_alpha_linear and rec_alpha_step==0:
        print(f"decode_alpha もしくは decode_alpha_max_epoch の指定に誤りがあったため decode_alpha_linear をオフにします")
        rec_alpha_linear = False
    if rec_rgb_alpha_linear and rec_rgb_alpha_max_epoch==0:
        print(f"decode_rgb_alpha もしくは decode_rgb_alpha_max_epoch の指定に誤りがあったため decode_rgb_alpha_linear をオフにします")
        rec_rgb_alpha_linear = False
    if rec_rgb_alpha_max_epoch>0:
        if args.decode_rgb_alpha is not None or args.decode_rgb_alpha!="":
            rec_rgb_alpha_def = [1.,1.,1.]
            rec_rgb_alpha_step = [0.,0.,0.]
            tmp = args.decode_rgb_alpha.replace(" ","").split(",")
            for s in tmp:
                color, alpha = s.split("=")
                if color == "r" or color == "R":
                    target_id = 0
                elif color=="g" or color == "G":
                    target_id = 1
                elif color=="b" or color == "B":
                    target_id = 2
                rec_rgb_alpha_def[target_id] = float(alpha)
                rec_rgb_alpha_step[target_id] = (float(alpha)-1.)/(rec_rgb_alpha_max_epoch*(len(dataset)/gradient_accumulation_steps))
    rec_rgb_alpha = rec_rgb_alpha_def[:]
    vae = load_vae_from_sd_checkpoint(model_file, dtype, use_xformers_flag)
    vae.to(device)
    #vae.requires_grad_(False)
    vae.eval()
    dataset.make_latent(vae)
    #########################################################################
    #いったん情報をまとめて表示
    print("\n[学習設定]\n==================================")
    print(f"学習セット： {data_dir}")
    print(f"ベースモデルファイル: {model_file}")
    print(f"epoch: {epoch} / pre epoch: {pre_epoch}")
    print(f"batch size: {batch_size}")
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    print(f"1epochあたりのstep数: {len(dataset)}")
    if gradient_accumulation_steps>1:
        print(f"実質的な1epochあたりのstep数(重みの更新回数): {int(len(dataset)//gradient_accumulation_steps)}")
    print(f"全体のstep数(重みの更新回数): {int(len(dataset)//gradient_accumulation_steps) * epoch}")
    print(f"max_data_loader_n_workers :{n_workers}")
    if args.seed is not None:
        print(f"seed: {args.seed}")
    print("---------------------------------")
    print(f"画像サイズ reso: {resolution}  min_reso: {min_resolution} 画像サイズの単位({divisible})")
    print(f"max_size: {max_size} min_size: {min_size}")
    print(f"optimizer: {optimizer_type} lr: {args.lr} warmup_init: {warmup_init}")
    print(f"decoderをmidとupで分割するかどうか: {decoder_param_split}")
    print(f"xformersの利用: {use_xformers_flag}")
    print(f"LPIPSのモデル名: {lpips_name}")
    print(f"discriminator_type: {discriminator_type}")
    print(f"discriminatorのネットワークサイズ: {discriminator_dim}")
    print("---------------------------------")
    print(f"latentのdropout率: {latent_dropout}")
    print(f"latentのdropoutを何epochまで続けるか: {max_latent_dropout_epoch}")
    print(f"latentに加えるノイズ強度: {latent_noise_rate}")
    print(f"decoderの出力補正: {rec_alpha}")
    if rec_alpha_step>0:
        print(f"decoderの出力補正を続けるepoch数: {rec_alpha_max} linear : {rec_alpha_linear}")
    print("----------------------------------")
    print(f"l2_lossの適用率: {l2_cost}")
    print(f"lpips_lossの適用率: {lpips_cost}")
    print(f"discriminatorの適用率: {disc_cost}")
    if max_grad_norm_flag:
        print(f"max_grad_norm: {max_grad_norm}")
    print("----------------------------------")
    print(f"出力ファイル名: {output_file}")
    if save_every_n_epoch > 0:
        print(f"epoch毎の保存: {save_every_n_epoch}")
    if enable_tensorboard:
        print(f"tensor board log dirctory: {tensorboard_output}")
        log_board = tensorboardX.SummaryWriter(tensorboard_output)
    print("==================================\n")
    #########################################################################
    if args.debug:
        #テスト出力
        data = dataset.__getitem__(0)
        test_latents = data["latents"].to(device=vae.device, dtype=vae.dtype)
        teacher = data["teacher"]
        test_img = (teacher / 2 + 0.5).clamp(0, 1)
        test_img = test_img.cpu().permute(0, 2, 3, 1).float().numpy()
        test_img = (test_img * 255).round().astype("uint8")
        test_img = [Image.fromarray(im) for im in test_img]
        test_img[0].save("test_0.png")

        gen_img(vae, test_latents, "test_1")
        vae.to("cpu")
        torch.cuda.empty_cache()#VRAM確保のために余計な情報は消去

        '''
        with torch.no_grad():
            test_img = vae.decode(test_latents).sample

        test_img = (test_img / 2 + 0.5).clamp(0, 1)
        test_img = test_img.cpu().permute(0, 2, 3, 1).float().numpy()
        test_img = (test_img * 255).round().astype("uint8")
        test_img = [Image.fromarray(im) for im in test_img]
        test_img[0].save("test_1.png")
        '''
    #########################################################################
    #latentを作成したのでとりあえずいったんVRAMを掃除
    vae.to("cpu")
    torch.cuda.empty_cache()#VRAM確保のために余計な情報は消去
    gc.collect()
    ######################## 学習に必要なものを作っていく
    #lpips
    print("///// ここのwarnigは無視していい //////")
    if lpips_name=="alex":
        loss_fn_lpips_loss = lpips.LPIPS(net='alex') # best forward scores
    else:
        loss_fn_lpips_loss = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
    loss_fn_lpips_loss.to(device=device)
    print("/////////////////////////////////////")

    #discriminator
    if discriminator_type=="simple":
        disc = SimpleDiscriminator(discriminator_dim, 1.75)
    elif discriminator_type=="compvis":
        disc = NLayerDiscriminator(discriminator_dim, 3)
    else:
        disc = Discriminator(discriminator_dim, 1)
    disc_opt = Adafactor(disc.parameters())
    disc_opt_sche = AdafactorSchedule(disc_opt)
    disc.to(device=device)
    disc.train()
    
    if optimizer_type == "8bitAdam":
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        except:
            print("8bit-Adamが取得できませんでした optimizer に Adafactor を設定します")
            optimizer_type = "Adafactor"
    
    if optimizer_type == "Adafactor":
        optimizer_class = Adafactor
        optimizer_arg["lr"] = None
        if warmup_init:
            optimizer_arg["warmup_init"] = True
    else:
        optimizer_class = torch.optim.AdamW

    parameters = []

    vae.post_quant_conv.to(device=device)
    #vae.post_quant_conv.requires_grad_(True)
    vae.post_quant_conv.train()
    parameters.append({"params": vae.post_quant_conv.parameters()})
    
    vae.decoder.to(device=device)
    #vae.decoder.requires_grad_(True)
    vae.decoder.train()
    if decoder_param_split:
        lr_names = ["post_quant_conv", "mid_block", "up_blocks"]
        parameters.append({"params": [p for n, p in vae.decoder.named_parameters() if (n.startswith("mid_block") or n.startswith("conv_in")) and p.requires_grad]})
        parameters.append({"params": [p for n, p in vae.decoder.named_parameters() if (not (n.startswith("mid_block") or n.startswith("conv_in"))) and p.requires_grad]})
    else:
        lr_names = ["post_quant_conv", "decoder", "decoder"]
        parameters.append({"params": vae.decoder.parameters()})

    optimizer = optimizer_class(parameters, **optimizer_arg)
    if optimizer_type == "Adafactor":
        initial_lr = args.lr if not args.lr==None else 0
        optimizer_scheduler = AdafactorSchedule(optimizer, initial_lr=initial_lr)
    else:
        optimizer_scheduler = get_constant_schedule(optimizer)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=n_workers, persistent_workers=persistent_data_loader_workers)
    
    datalen = len(dataset)
    pre_total_step = pre_epoch * datalen
    total_step = epoch * datalen
    #先にdiscriminatorをある程度学習させておく
    print("\n======================\n[ Pre Train / Discriminatorだけ先に少し学習をしておく ]\n======================\n")
    with tqdm.tqdm(range(pre_total_step), desc="steps") as pbar:
        global_step = 0
        for i in range(pre_epoch):
            fake_losse_sum = 0.
            real_losse_sum = 0.
            pbar.set_description("[Epoch %d] " % i)
            for step, data in enumerate(train_dataloader):
                latents = data["latents"].to(device=device, dtype=dtype)
                teacher = data["teacher"].to(device=device, dtype=dtype)
                #####################################################
                #discriminator
                with torch.no_grad():
                    reconstructions = vae.decode(latents).sample
                loss_fake, loss_real = disc.loss_func_discriminator(reconstructions, teacher)
                #dis_fake = disc(reconstructions)
                #dis_real = disc(teacher)
                #loss_fake = F.softplus(dis_fake).mean()
                #loss_real = F.softplus(-dis_real).mean()
                #loss_fake = torch.nn.functional.mse_loss(dis_fake, torch.zeros_like(dis_fake))
                #loss_real = torch.nn.functional.mse_loss(dis_real, torch.ones_like(dis_real))
                loss_disc = loss_fake + loss_real
                disc_opt.zero_grad()
                loss_disc.backward()
                now_loss = loss_fake.detach().clone().item()
                fake_losse_sum += now_loss
                if enable_tensorboard:
                    log_board.add_scalar("discriminator/fake_loss", now_loss, global_step)
                now_loss = loss_real.detach().clone().item()
                real_losse_sum += now_loss
                if enable_tensorboard:
                    log_board.add_scalar("discriminator/real_loss", now_loss, global_step)
                pbar.set_postfix({"loss_fake": f"{fake_losse_sum/(step+1):.4f}", "loss_real": f"{real_losse_sum/(step+1):.4f}"})
                del loss_disc, loss_fake, loss_real
                torch.cuda.empty_cache()
                disc_opt.step()
                disc_opt_sche.step()
                #del dis_fake, dis_real
                torch.cuda.empty_cache()
                #####################################################
                global_step += 1
                pbar.update(1)
            print(f"{print_command.DEL}[Epoch {i}] loss_fake: {fake_losse_sum/(step+1):.4f} loss_real: {real_losse_sum/(step+1):.4f}")
    print("\n======================\n[ Train / 本番の学習　Decoderの学習 ]\n======================\n")
    with tqdm.tqdm(range(total_step), desc="steps") as pbar:
        global_step = 0
        train_step = 0
        if latent_dropout<=0.:
            dropout_func = lambda x: x
        else:
            print("\nset dropout func")
            dropout_func = torch.nn.Dropout(latent_dropout).to(device=vae.device, dtype=vae.dtype)
        for i in range(epoch):
            epst = time.time()
            loss_sum = 0.
            fake_losse_sum = 0.
            real_losse_sum = 0.
            loss_vae_sum = 0.
            loss_fake_sum = 0.
            loss_lpips_sum = 0.
            logs = {}
            if max_latent_dropout_epoch <= i:
                dropout_func = lambda x: x
            pbar.set_description("[Epoch %d] " % i)
            for step, data in enumerate(train_dataloader):
                if rec_alpha_linear:
                    rec_alpha = rec_alpha_def - (train_step * rec_alpha_step)
                if rec_alpha_max-i <= 0: rec_alpha = 1.
                ####################################################
                if len(rec_rgb_alpha) == 3:
                    for _i in range(3):
                        if rec_rgb_alpha_linear:
                            rec_rgb_alpha[_i] = rec_rgb_alpha_def[_i] - (train_step * rec_rgb_alpha_step[_i])
                        if rec_rgb_alpha_max_epoch-i <= 0:
                            rec_rgb_alpha[_i] = 1.
                ####################################################
                latents = data["latents"].to(device=device, dtype=dtype)
                if latent_noise_rate>0:
                    latent_noise = torch.normal(mean=0, std=1, size=latents.size()) * latent_noise_rate
                    latents = latents + latent_noise.to(device=device, dtype=dtype)
                latents = dropout_func(latents)
                teacher = data["teacher"].to(device=device, dtype=dtype)
                #####################################################
                #discriminator
                with torch.no_grad():
                    reconstructions = vae.decode(latents).sample
                disc.requires_grad_(True)
                loss_fake, loss_real = disc.loss_func_discriminator(reconstructions, teacher)
                #dis_fake = disc(reconstructions)
                #dis_real = disc(teacher)
                #loss_fake = F.softplus(dis_fake).mean()
                #loss_real = F.softplus(-dis_real).mean()
                #loss_fake = torch.nn.functional.mse_loss(dis_fake, torch.zeros_like(dis_fake))
                #loss_real = torch.nn.functional.mse_loss(dis_real, torch.ones_like(dis_real))
                loss_disc = loss_fake + loss_real
                disc_opt.zero_grad()
                loss_disc.backward()
                now_loss = loss_fake.detach().clone().item()
                fake_losse_sum += now_loss
                now_loss = loss_real.detach().clone().item()
                real_losse_sum += now_loss
                del loss_disc, loss_fake, loss_real
                torch.cuda.empty_cache()
                disc_opt.step()
                disc_opt_sche.step()
                #del dis_fake, dis_real
                torch.cuda.empty_cache()
                #####################################################
                #learning
                z = vae.post_quant_conv(latents)
                reconstructions = ((vae.decoder(z) + rec_alpha_blight) * rec_alpha) - rec_alpha_blight
                
                if len(rec_rgb_alpha) == 3:
                    for _i in range(3):
                        reconstructions[:, _i] = ((reconstructions[:, _i] + rec_rgb_alpha_blight) * rec_rgb_alpha[_i]) - rec_rgb_alpha_blight

                loss_vae = torch.nn.functional.mse_loss(reconstructions, teacher, reduction="none").mean()
                #loss = torch.mean(torch.square(reconstructions - teacher))

                #discriminator loss
                disc.requires_grad_(False)
                loss_fake = disc.loss_func(reconstructions, teacher)
                #dis_fake = disc(reconstructions)
                #loss_fake = F.softplus(-dis_fake).mean()
                #loss_fake = torch.nn.functional.mse_loss(dis_fake, torch.ones_like(dis_fake)).mean()

                #lpips loss
                loss_fn_lpips_loss.requires_grad_(False)
                loss_lpips = loss_fn_lpips_loss(teacher, reconstructions).mean()
                
                loss = (loss_vae * l2_cost) + (loss_fake * disc_cost) + (loss_lpips * lpips_cost)

                now_loss = loss.detach().clone().item()
                loss_sum += now_loss

                loss /= gradient_accumulation_steps

                loss.backward()

                now_loss = loss_vae.detach().clone().item()
                loss_vae_sum += now_loss
                if enable_tensorboard:
                    log_board.add_scalar("decoder/MSE_loss", now_loss, global_step)
                now_loss = loss_fake.detach().clone().item()
                loss_fake_sum += now_loss
                if enable_tensorboard:
                    log_board.add_scalar("decoder/Discriminator_loss", now_loss, global_step)
                now_loss = loss_lpips.detach().clone().item()
                loss_lpips_sum += now_loss
                if enable_tensorboard:
                    log_board.add_scalar("decoder/LPIPS_loss", now_loss, global_step)
                
                logs["loss"]= f"{loss_sum / (step+1):.4f}"
                pbar.set_postfix(logs)

                del loss, loss_fake, loss_vae, loss_lpips#, dis_fake
                torch.cuda.empty_cache()
                if (step+1)%gradient_accumulation_steps==0:
                    if max_grad_norm_flag:
                        torch.nn.utils.clip_grad_norm_(vae.decoder.parameters(), max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(vae.post_quant_conv.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer_scheduler.step()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    if enable_tensorboard:
                        log_board.add_scalar("decoder_alpha/decode_alpha", rec_alpha, global_step)
                        if len(rec_rgb_alpha) == 3:
                            for _i in range(3):
                                log_board.add_scalar(f"decoder_alpha/decode_{rec_rgb_name[_i]}_alpha", rec_rgb_alpha[_i], global_step)
                        for lr_num, now_lr in enumerate(optimizer_scheduler.get_lr()):
                            log_board.add_scalar(f"decoder_lr/lr_{lr_names[lr_num]}", now_lr, global_step)
                    train_step += 1
                #####################################################
                global_step += 1
                pbar.update(1)
            epoch_time = time.time() - epst
            now_epoch_time = epoch_time
            epoch_time_log = ""
            for t in range(2):
                et = now_epoch_time // (60**(2-t))
                epoch_time_log = f"{epoch_time_log}:{int(et):02d}"
                now_epoch_time = now_epoch_time % (60**(2-t))
            epoch_time_log = f"{epoch_time_log}:{now_epoch_time:02.02f} ({datalen/epoch_time:02.03f} step/time)"
            print(f"{print_command.DEL}[Epoch {i}]  estime: {epoch_time_log} (loss: {loss_sum/(step+1):.6f})")
            print(f"discriminator: loss_fake: {fake_losse_sum/(step+1):.4f} loss_real: {real_losse_sum/(step+1):.4f}")
            print(f"VAE decoder: MSEloss: {loss_vae_sum/(step+1):.4f} loss_dis: {loss_fake_sum/(step+1):.4f} loss_lpips: {loss_lpips_sum/(step+1):.4f}")
            if save_every_n_epoch>0 and (i+1)<epoch:
                if (i+1)%save_every_n_epoch==0:
                    save_vae(vae.state_dict(), f"{args.output_file}_epoch{i:03d}.{args.save_type}", args.save_type)

    if args.debug:
        gen_img(vae, test_latents, "test_2")
        '''
        with torch.no_grad():
            test_img = vae.decode(test_latents).sample

        test_img = (test_img / 2 + 0.5).clamp(0, 1)
        test_img = test_img.cpu().permute(0, 2, 3, 1).float().numpy()
        vae.to("cpu")
        torch.cuda.empty_cache()#VRAM確保のために余計な情報は消去
        test_img = (test_img * 255).round().astype("uint8")
        test_img = [Image.fromarray(im) for im in test_img]
        test_img[0].save("test_2.png")
        '''
    vae.to("cpu")
    torch.cuda.empty_cache()#VRAM確保のために余計な情報は消去
    save_vae(vae.state_dict(), output_file, args.save_type)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="yamlファイルから設定を読み込む場合に使う 拡張子不要 学習終了後に設定項目は <output_file>_日付_時分.yamlに保存される")
    parser.add_argument("--dataset", type=str, default=None, help="学習に使うデータセットのディレクトリ　階層は下の方まで探索する　指定しなければ起動時にダイアログで設定できる")
    parser.add_argument("--model", type=str, default=None, help="学習に使うベースとなるvaeファイル 指定しなければ起動時いダイアログで設定できる")

    parser.add_argument("--epoch", type=int, default=10, help="epoch数")
    parser.add_argument("--pre_epoch", type=int, default=1, help="Decoderの学習前にあらかじめDiscriminatorの学習を回しておく epoch数")
    parser.add_argument("--batch", type=int, default=1, help="batchサイズ")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="バッチを大きくする代わりの仕組み VRAM節約のための仕組み 実質的なバッチサイズ = batch * これ")
    parser.add_argument("--not_shuffle", action="store_false", help="データセットの順番を維持したまま学習したい時用")
    parser.add_argument("--val_rate", type=float, default=0., help="[WIP]データセットを評価用と学習用に分割する時に使う・大規模学習じゃないならいらない気もする")
    parser.add_argument("--seed", type=int, default=None, help="学習の乱数固定")

    parser.add_argument("--resolution", type=str, default="256,256", help="画像の最大サイズ")
    parser.add_argument("--min_resolution", type=str, default="128,128", help="画像の最小サイズ")
    parser.add_argument("--max_size", type=int, default=512, help="画像の最大幅 長辺の最大サイズ")
    parser.add_argument("--min_size", type=int, default=128, help="画像の最小幅 短辺の最小サイズ")
    parser.add_argument("--divisible", type=int, default=64, help="画像サイズの単位 8の倍数 VAEのみ学習するので8でも問題ないけど実際のところどうなのかはわからない")
    parser.add_argument("--bucket_serch_step", type=int, default=1, help="画像のアス比に近くなるサイズを探索する範囲　値を大きくするほどアス比は最適化されるが拡大縮小幅も大きくなる")
    parser.add_argument("--make_clipping", type=float, default=0., help="指定した割合よりも縮小する場合ランダムにクリッピングした教師データも追加する 0もしくは1以上で無効化")
    parser.add_argument("--make_clip_num", type=int, default=1, help="make_clippingで水増しするデータ数")

    parser.add_argument("--optimizer", type=str, default="Adafactor", help="[AdamW|8bitAdam|Adafactor] のどれか選べる")
    parser.add_argument("--lr", type=float, default=None, help="Adafactorの場合 initial_lrとして設定される")
    parser.add_argument("--warmup_init", action="store_true", help="Adafactorのwarmup_initを有効にするかどうか")
    parser.add_argument("--grad_clip", type=float, default=1., help="学習しすぎるのを防ぐための値　小さいほうが防ぐけどつまり学習の進みを遅くする仕組みでもある 0以下指定で無効化")
    parser.add_argument("--decoder_param_split", action="store_true", help="デコーダーのパラメータをmidとupで分割してLr調整をするか Adafactor限定")

    parser.add_argument("--decode_alpha", type=float, default=1., help="decoderの出力に掛ける補正値")
    parser.add_argument("--decode_alpha_max_epoch", type=int, default=0, help="decoder alphaを適用するepoch数 0だと無効化 ")
    parser.add_argument("--decode_alpha_linear", action="store_true", help="decode_alpha_max_epochの指定を使った時 直線的にdecode_alphaの値を1.0に変化させていく")
    parser.add_argument("--decode_alpha_blight", action="store_true", help="alpha値の計算を0を起点としたものにする 通常の計算式の場合正規化された値なので強さというよりコントラストに近い")
    parser.add_argument("--decode_rgb_alpha", type=str, default=None, help="RGB毎にdecoderの出力にalpha値を乗算する r＝0.5,b＝1.0 等のように指定する（＝は半角　説明文が全角なのはhelpが半角＝だとエラーになるから）")
    parser.add_argument("--decode_rgb_alpha_max_epoch", type=int, default=0, help="RGB毎にdecoderno出力にalphaを適用する最大epoch数 0だと無効化 RGB毎に設定はできないので注意")
    parser.add_argument("--decode_rgb_alpha_linear", action="store_true", help="RGB毎に適用するalpha値をmax_epochで指定したepochまで直線的にalphaの値を1.0にしていく")
    parser.add_argument("--decode_rgb_alpha_blight", action="store_true", help="RGB値に対するalpha値の計算を0を起点としたものにする 通常の計算式の場合正規化された値なので強さというよりコントラストに近い")
    parser.add_argument("--l2_cost", type=float, default=1., help="l2 lossの補正値 教師画像と見比べた時に実際の違いに対しての注目度みたいなもの")
    parser.add_argument("--lpips_cost", type=float, default=0.5, help="lpips loss の補正値 LPIPSに教師画像と生成画像を見比べさせた時の評価値に対する注目度みたいなもの")
    parser.add_argument("--discriminator_cost", type=float, default=0.5, help="discriminator loss の補正値 Discriminatorに生成画像を見せた時に本物と騙されてる程度に対する注目度みたいなもの")

    parser.add_argument("--discriminator_type", type=str, default="def", help="discriminatorのタイプ defで通常 simpleで非常にシンプルなdiscriminatorを使う[def|simple|compvis]")
    parser.add_argument("--discriminator_dim", type=int, default=64, help="discriminatorのネットワークサイズ 大きければいいってわけでもない気がするけど適切なサイズいまいちわからない")
    parser.add_argument("--not_use_xformers", action="store_false", help="xformersを使いたくない時用")
    parser.add_argument("--latent_dropout", type=float, default=0., help="decoder入力前のlatentにdropoutを適用する SDのVAEのようなモデルで使う仕組みじゃない気がするけど")
    parser.add_argument("--max_latent_dropout_epoch", type=int, default=0, help="latent dropout を適用するepoch数 0なら最後まで適用")
    parser.add_argument("--latent_noise_rate", type=float, default=0., help="latentにノイズを加える混ぜ具合 1.0なら100パーセント 0なら0パーセント(無効化)")

    parser.add_argument("--lpips_name", type=str, default="alex", help="画像評価モデルのLPIPSに使うモデル名 [alex|vgg]")
    parser.add_argument("--max_data_loader_n_workers", type=int, default=1, help="DataLoaderの数 VAE学習自体は軽いから適当に増やしていいかもしれないCPUに合わせて上限を超えて設定しても勝手に調整する")

    parser.add_argument("--output_file", type=str, default="vae_output\\testvae", help="出力ファイル名　ディレクトリから指定できる ディレクトリがなかったら作る　拡張子は自動で補完するからいらない")
    parser.add_argument("--save_type", type=str, default="safetensors", help="保存形式[pt|safetensors]")
    parser.add_argument("--save_every_n_epoch", type=int, default=0, help="指定したepoch毎に途中経過を保存する 0なら無効")

    parser.add_argument("--debug", action="store_true", help="現状では単純に縮小した元画像　学習前のDecoder出力 学習後のDecoder出力 出すだけ")
    parser.add_argument("--enable_tensorboard", action="store_true", help="学習の経過情報に関する記録をtensorboardを使って出力する")
    args = parser.parse_args()
    
    #yamlをコンフィグとして読み込む場合 読み込み前でargs.configを使ってるから読み込み時のyamlファイルは何の影響もない
    if (args.config is not None) and (not args.config==""):
        if os.path.splitext(args.config)[-1] == ".yaml":
            args.config = os.path.splitext(args.config)[0]
        config_name = args.config
        config_path = f"{args.config}.yaml"
        _output_file = ""
        if os.path.exists(config_path):
            if not args.output_file == "vae_output\\testvae":
                _output_file = args.output_file
            print(f"{config_path} から設定を読み込み中...")
            margs, rest = parser.parse_known_args()
            with open(config_path, mode="r") as f:
                configs = yaml.unsafe_load(f)
            args_names = argparse.Namespace(**configs)
            args = parser.parse_args(args=rest, namespace=args_names)
            args.config = config_name
            if not _output_file == "":
                args.output_file = _output_file
            if args.lr is not None:
                args.lr = float(args.lr)
        else:
            print(f"{config_path} が見つかりませんでした")
    
    train(args)

    #学習が終わったら現在のargsを保存する
    _t = datetime.datetime.today().strftime('%Y%m%d_%H%M')
    config_name = f"vae_config_{os.path.basename(args.output_file)}_{_t}.yaml"
    print(f"{config_name} に設定を書き出し中...")
    with open(config_name, mode="w") as f:
        yaml.dump(args.__dict__, f, indent=4)
    print("done!")
