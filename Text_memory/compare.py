from collections import defaultdict
import numpy as np
import os
import torch.optim as optim
import clip
from tqdm import tqdm 
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from PIL import Image
import yaml
import re
import glob
import cv2
import matplotlib.pyplot as plt

def read_with_while(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    
    category_list = data['category']
    category_list = data['abnormal_list']
    return category_list, abnormal_list

def text_to_dict(text):
    result = {}
    items = text.split(',')
    for item in items:
        item = item.strip()
        if not item:
            continue
        parts = item.split()
        if len(parts) < 1:
            continue
        key = parts[-1]
        quantity_parts = parts[:-1]
        if not quantity_parts:
            continue
        quantity_str = ' '.join(quantity_parts).lower()
        quantity = int(quantity_str)
        result[key] = quantity
    return result

def process_masks(txt_folder, png_folder, target_str):
    mask = Image.new('L', (448, 448), 0)
    for filename in os.listdir(txt_folder):
        txt_path = os.path.join(txt_folder, filename)
        if txt_path.endswith(".txt"):
            with open(txt_path, 'r') as f:
                content = f.read().replace('\n', '')
                        
            if content == target_str:
                base_name = os.path.splitext(os.path.basename(txt_path))[0]
                png_path = os.path.join(png_folder, f"{base_name}.png")
                
                if os.path.exists(png_path):
                    img = Image.open(png_path)
                    img = img.resize((448, 448)).convert('L')
                    mask.paste(img)
                else:
                    print(f"Not found {png_path}")
        
    return mask 

def process_masks2(txt_folder, png_folder, target_str):
    mask = Image.new('L', (448, 448), 0)
    mask_path = f"/root/TMUAD/Text_memory/train/mask-class/{target_str}.png"
    img = Image.open(mask_path)
    img = img.resize((448, 448)).convert('L')
    mask.paste(img)    
    return mask  

def process_masks3(txt_folder, png_folder, target_str):
    mask = Image.new('L', (448, 448), 0)
    mask_path = f"/root/TMUAD/Text_memory/train/mask-class/{target_str}.png"
    img = Image.open(mask_path)
    img = img.resize((448, 448)).convert('L')
    mask.paste(img)    
    for filename in os.listdir(txt_folder):
        txt_path = os.path.join(txt_folder, filename)
        if txt_path.endswith(".txt"):
            with open(txt_path, 'r') as f:
                content = f.read().replace('\n', '')            
            base_name = os.path.splitext(os.path.basename(txt_path))[0]
            png_path = os.path.join(png_folder, f"{base_name}.png")
            
            if os.path.exists(png_path):
                img = Image.open(png_path)
                img = img.resize((448, 448)).convert('L')
                mask1_arr = np.array(mask)
                mask2_arr = np.array(img)

                mask2_white = mask2_arr == 255
                mask1_arr[mask2_white] = 0

                result = Image.fromarray(mask1_arr)
                mask = result
            else:
                print(f"Not found {png_path}")
        
    return mask  

def compare_dicts(dict1, dict2):
    differences = []
    all_keys = set(dict1.keys()) | set(dict2.keys())
    for key in all_keys:
        val1 = dict1.get(key, None)
        val2 = dict2.get(key, None)
        if val1 != val2:
            differences.append((key, val1, val2))
    return differences

def format_differences(differences,idx,category):
    txtdir_path = f"/root/TMUAD/Text_memory/test/{is_abnormal}/{category}/{idx:03d}"
    pngdir_path = f"/root/TMUAD/masks/{category}/test/{is_abnormal}/{idx:03d}"
    if not differences:
        with open(f"/root/TMUAD/Text_memory/similar/{category}-{is_abnormal}.txt", 'a', encoding='utf-8') as f:
            f.write("1\n")
        mask_is_abnormal = Image.new('L', (448, 448), 0)
        os.makedirs(f"/root/TMUAD/Text_memory/text_anomaly_mask/test/{is_abnormal}/{category}",exist_ok=True)
        mask_is_abnormal.save(f"/root/TMUAD/Text_memory/text_anomaly_mask/test/{is_abnormal}/{category}/{idx}.png")
        zero_tensor = torch.zeros(1, 1, 448, 448)
        os.makedirs(f"/root/TMUAD/Text_memory/text_anomaly/test/{is_abnormal}/{category}",exist_ok=True)
        torch.save(zero_tensor, f'/root/TMUAD/Text_memory/text_anomaly/test/{is_abnormal}/{category}/{idx}.pt')
        return "Same"
    messages = []
    with open(f"/root/TMUAD/Text_memory/similar/{category}-{is_abnormal}.txt", 'a', encoding='utf-8') as f:
        f.write("0\n")
    mask_all = Image.new('L', (448, 448), 0)
    for key, val1, val2 in differences:
        
        if val1 is None:
            messages.append(f"Lack '{key}'")
            mask_generator = process_masks2(txtdir_path, pngdir_path, key)
            mask_all.paste(mask_generator)
        elif val2 is None:
            messages.append(f"Excess'{key}'")
            mask_generator = process_masks(txtdir_path, pngdir_path, key)
            mask_all.paste(mask_generator)
        else:
            if val1 > val2:
                mask_generator = process_masks(txtdir_path, pngdir_path, key)
                mask_all.paste(mask_generator)
                messages.append(f"'{key}' over {val1-val2}")
            else:
                mask_generator = process_masks3(txtdir_path, pngdir_path, key)
                mask_all.paste(mask_generator)
                messages.append(f"'{key}' lack {val2-val1}")
    os.makedirs(f"/root/TMUAD/Text_memory/text_anomaly/test/{is_abnormal}/{category}",exist_ok=True)
    os.makedirs(f"/root/TMUAD/Text_memory/text_anomaly_mask/test/{is_abnormal}/{category}",exist_ok=True)
    mask_all.save(f"/root/TMUAD/Text_memory/text_anomaly_mask/test/{is_abnormal}/{category}/{idx}.png")
    mask_array = np.array(mask_all, dtype=np.float32) / 255.0
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)
    torch.save(mask_tensor, f'/root/TMUAD/Text_memory/text_anomaly/test/{is_abnormal}/{category}/{idx}.pt')
    return "".join(messages)
import difflib

def read_and_split_b(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.readlines()
    
    split_set = []
    for line in text:
        parts = line.strip().split(',')  
        split_set.append(line)     
    return split_set

def compare_txt(a_file_path, b_file_path, category):
    with open(a_file_path, 'r', encoding='utf-8') as file1:
        for idx,line_a in enumerate(file1):
            max_similarity = 0
            sim_line = ''
            if line_a == '\n':
                continue
            with open(b_file_path, 'r', encoding='utf-8') as file2:
                for line_b in file2:
                    similarity = difflib.SequenceMatcher(None, line_a.strip(), line_b.strip()).ratio()
                    if similarity > max_similarity:
                        max_similarity = similarity
                        sim_line = line_b
            dict1 = text_to_dict(line_a)
            dict2 = text_to_dict(sim_line)
            differences = compare_dicts(dict1, dict2)
            answer_txt = format_differences(differences,idx,category,is_abnormal)
            with open(f"/root/TMUAD/Text_memory/similar/text/{category}-{is_abnormal}.txt", 'a', encoding='utf-8') as f:
                f.write(answer_txt+"\n")
            print(answer_txt)

def main():
    category_list, abnormal_list = read_with_while("/root/TMUAD/config/compare.yaml")
    for category in category_list:
        for is_abnormal in abnormal_list:
            b_file_path = f'/root/TMUAD/Text_memory/train/{category}.txt'
            a_file_path = f'/root/TMUAD/Text_memory/test/{is_abnormal}/{category}.txt'
            compare_txt(a_file_path, b_file_path, category, is_abnormal)


if __name__ == '__main__':
    main()