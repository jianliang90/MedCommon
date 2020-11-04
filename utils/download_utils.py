import os
import pandas as pd
import requests
from tqdm import tqdm

import csv

import fire



def download_dcm(Down_path, series_ID, down_dir):
    series_folder = os.path.join(Down_path, series_ID)
    if not os.path.exists(series_folder):
        os.makedirs(series_folder)
    conts = down_dir.split(".dcm")
    temp_name = conts[0].split("/")[-1] + ".dcm"
    
    f = requests.get(down_dir)
    write_name = os.path.join(series_folder, temp_name)
    with open(write_name,"wb") as code:
        code.write(f.content)
        # print(os.path.basename(write_name))

    files = os.listdir(Down_path)


def download_dcms_with_website(download_pth, config_file):
    '''
    invoke cmd: python common/utils/download_utils.py download_dcms_with_website 'pulmonary_embolism/data/pulmonaryEmbolism/data_batch_1/images' '../data/pulmonaryEmbolism/data_batch_1/文件内网地址信息-导出结果.xlsx'
    '''
    continue_flag = True
    sheet_num = 0
    while continue_flag == True:
        try:
            df = pd.read_excel(config_file, sheet_name = sheet_num, header = [0])
            for i in tqdm(range(len(df))):
                row = list(df.iloc[i,:].values)
                download_dcm(download_pth, row[0], row[3])
            sheet_num = sheet_num +1
        except:
            continue_flag = False



def download_label(Down_path, series_IDs, down_dirs):
    if not os.path.exists(Down_path):
        os.makedirs(Down_path)
    files = os.listdir(Down_path)
    # if len(files) == 0:
    if True:
        assert(len(series_IDs) == len(down_dirs))
        for i in range(len(series_IDs)):
            temp_ID = series_IDs[i]
            down_addr = down_dirs[i]
            try:
                xx = len(down_addr)
            except:
                continue
            f=requests.get(down_addr)
            temp_name = os.path.join(Down_path, '{}.mha'.format(temp_ID))

            with open(temp_name,"wb") as code:
                code.write(f.content)
                print(os.path.basename(temp_name))

def download_mha_with_csv(download_path, config_file):
    '''
    python common/utils/download_utils.py download_mha_with_csv '../data/pulmonaryEmbolism/data_batch_1/masks' '../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2706.csv'
    '''
    # label_info_file = '../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2706.csv'
    # Down_path = '../data/pulmonaryEmbolism/data_batch_1/masks'
    # label_info_file = '../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2899.csv'
    # Down_path = '../data/pulmonaryEmbolism/data_batch_1/masks'
    os.makedirs(download_path, exist_ok=True)
    data = pd.read_csv(config_file)
    series_ids = list(data.iloc[:,5].values)
    urls = list(data.iloc[:,15].values)
    download_label(download_path, series_ids, urls)


def get_series_uids(infile, column_name, outfile=None):
    '''

    note: 数仓在做标注的时候会导出一张标注的表格，用以记录序列和mask的对应关系。该函数的作用是根据这张表格，找到原始序列的uid。该uid后续需要获取内网地址进行下载。

    get_series_uids('../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2706.csv', '序列编号', '../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2706.txt')
    get_series_uids('../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2899.csv', '序列编号', '../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2899.txt')
    invoke cmd: python download_pulmonary_embolism_images.py get_series_uids '../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2706.csv' '序列编号' '../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2706.txt'
    invoke cmd: python download_pulmonary_embolism_images.py get_series_uids '../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2899.csv' '序列编号' '../data/pulmonaryEmbolism/data_batch_1/image_anno_TASK_2899.txt'
    '''
    df = pd.read_csv(infile)
    series_uids = list(set(df[column_name].tolist()))
    if outfile is not None:
        with open(outfile, 'w') as f:
            f.write('\n'.join(series_uids))
    return series_uids


if __name__ == '__main__':
    fire.Fire()