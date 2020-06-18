import csv


###########################################################################
def read_csv(file):
    """
    read csv
    :param file:
    :return:
    """
    obj = list()
    f = open(file, 'rt')
    try:
        reader = csv.reader(f)
        for row in reader:
            obj.append(row)
    finally:
        f.close()

    return obj


###########################################################################
def write_csv(row_list, file):
    """
    write csv
    :param row_list:
    :param file:
    :return:
    """
    f = open(file, 'wt')
    try:
        writer = csv.writer(f)
        for row in row_list:
            writer.writerow(row)
    finally:
        f.close()
