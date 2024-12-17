import csv

def transpose_file(input_file, output_file):

    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        data = list(reader)
        
        transposed_data = list(zip(*data))
    
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(transposed_data)

transpose_file('./data/waimea/P-annotated.csv', './data/waimea/P-annotated.csv')
