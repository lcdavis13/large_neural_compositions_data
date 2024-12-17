import csv

def remove_header_and_key(input_file, output_file):

    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        data = list(reader)
        
        cleaned_data = [row[1:] for row in data[1:]]
    
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(cleaned_data)

remove_header_and_key('./data/waimea/P-annotated.csv', './data/waimea/P.csv')
