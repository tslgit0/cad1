import csv
csvFile=open("/home/cad429/code/panda/csv/try.csv",'a')
list=['name']
writer=csv.writer(csvFile)

writer.writerow(list)
writer.writerow(list)
csvFile.close()
