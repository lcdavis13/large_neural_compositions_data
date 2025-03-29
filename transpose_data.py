import pandas as pd

fname = "./data/synth/gLV__N5000_c0.05_nu1.0/P-original.csv"
data = pd.read_csv(fname, header=None)
print(data.shape)

def transpose_large_data(df):
    transposed_chunks = []
    for _, row in df.iterrows():
        transposed_chunks.append(row.to_frame().T)
        if len(transposed_chunks) > 1000:
            transposed_df = pd.concat(transposed_chunks)
            transposed_chunks = []
    if transposed_chunks:
        transposed_df = pd.concat(transposed_chunks)
    return transposed_df


transposed_data = transpose_large_data(data)
print(transposed_data.shape)

transposed_data.to_csv(fname, header=False, index=False)

print("Done")
