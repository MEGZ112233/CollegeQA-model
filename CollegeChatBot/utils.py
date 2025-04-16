import csv

def upload_chunks(data_path):
    chunks = []
    with open(data_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[0]:
                chunks.append(row[0])
    return chunks