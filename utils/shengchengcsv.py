import csv
def csv_generator(input):
    csvFile=open("/home/cad429/code/panda/csv/try.csv",'a')
    writer=csv.writer(csvFile)

    writer.writerow(input)

    csvFile.close()
