image_id = []
image_path = []
with open('C:\\Datasets\\VGGface2\\test.csv', 'r') as f:
    for row in f:
        row = row.replace('\n', '')
        id = row.split('/')[0]
        path = row
        image_id.append(id)
        image_path.append(path)

print(image_path)
