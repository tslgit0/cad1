import csv
import os
traincsv_path='/home/cad429/code/data/train1.csv'
maskcsv_path='/home/cad429/code/data/mask.csv'
path='/home/cad429/code/data'
def generate_csv(path,type,csv_path):
    with open(csv_path,'w',newline='') as csvfile:
        svwriter=csv.writer(csvfile,dialect="excel")
        svwriter.writerow(['name'])
        listdir=os.listdir(os.path.join(path,type))
        for i in listdir:
            j=i.split('.')[0]
            #print(j)
            svwriter.writerow([j])

def generate_csv1(path,type,csv_path):
    with open(csv_path,'w',newline='') as csvfile:
        svwriter=csv.writer(csvfile,dialect="excel")
        svwriter.writerow(['name'])
        listdir=os.listdir(os.path.join(path,type))
        for i in listdir:
            j=i.split('.')[0].split('_')[0]
            #print(j)
            svwriter.writerow([j])

generate_csv(path,'train_images',traincsv_path)
generate_csv1(path,'train_label_masks',maskcsv_path)