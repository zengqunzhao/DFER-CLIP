from glob import glob
import os

def update(file, old_str, new_str):
    file_data = ""
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if old_str in line:
                line = line.replace(old_str,new_str)
            file_data += line
    with open(file, "w", encoding="utf-8") as f:
        f.write(file_data)

old_path = "/data/EECS-IoannisLab/datasets/DFEW/DFEW_Frame_Face/"
new_path = "/put your data path here"
all_txt_file = glob(os.path.join('DFEW_*.txt'))
for txt_file in all_txt_file:
    update(txt_file, old_path, new_path)

old_path = "/data/EECS-IoannisLab/datasets/FERV39K/"
new_path = "/put your data path here"
all_txt_file = glob(os.path.join('FERV39K_*.txt'))
for txt_file in all_txt_file:
    update(txt_file, old_path, new_path)
    
old_path = "/data/EECS-IoannisLab/datasets/MAFW/"
new_path = "/put your data path here"
all_txt_file = glob(os.path.join('MAFW_*.txt'))
for txt_file in all_txt_file:
    update(txt_file, old_path, new_path)