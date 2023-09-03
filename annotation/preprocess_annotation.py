import os
import pandas as pd
from glob import glob

DFEW_SOURCE = "/home/zonepg/datasets/DFEW"
DFEW_CLIP_SOURCE = os.path.join(DFEW_SOURCE, "Clip", "clip_224x224")

def main():
    all_anno_file = sorted(glob(os.path.join(DFEW_SOURCE, "EmoLabel_DataSplit", "**", "*.csv")))
    for anno_file in all_anno_file:
        print(anno_file)
        data = pd.read_csv(anno_file)
        write_data = []
        write_file_path = "DFEW_{}_{}.txt".format(anno_file.split("/")[-1].split(".")[0], anno_file.split("/")[-2])
        write_file_path = os.path.join("annotation", write_file_path)
        for index, row in data.iterrows():
            video_name = str(row["video_name"]).zfill(5)
            full_video_name = os.path.join("/data/EECS-IoannisLab/datasets/DFEW/DFEW_Frame_Face", video_name)
            label = str(row["label"] - 1)
            clip_folder_path = os.path.join(DFEW_SOURCE, "Clip", "clip_224x224", video_name)
            clip_image_path = sorted(glob(os.path.join(clip_folder_path, "*.jpg")))
            frame_num = len(clip_image_path)
            write_data.append("{} {} {}\n".format(full_video_name, frame_num, label))
        with open(write_file_path, "w") as f:
            f.writelines(write_data)




if __name__ == "__main__":
    main()
