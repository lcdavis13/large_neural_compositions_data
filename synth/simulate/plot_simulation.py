import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


otu_num = 256
proj = f"{otu_num}/random-1-RepHeun"
# proj = f"{otu_num}/random-1-RepDEQ"
# proj = f"{otu_num}/random-1-gLV"
# proj = f"{otu_num}/random-1-Rep"
# proj = f"{otu_num}/random-weak-Rep"
# proj = f"{otu_num}/random-weak-gLV"
# proj = f"{otu_num}/random-noBias-Rep"
# proj = f"{otu_num}/random-noBias-gLV"
# proj = f"{otu_num}/cnode1-1k-Rep"
# proj = f"{otu_num}/cnode1-100k-Rep"
# proj = f"{otu_num}/cnode1-1k-gLV"
# proj = f"{otu_num}/cnode1-100k-gLV"
inpath = f"synth/simulate/debug/{proj}/"
mask_file = f"synth/_data/{otu_num}/_binary_0.csv"

# File paths
num_otus = 256
# phylo = f"{num_otus}@random"
# taxonomic_level = f"{num_otus}@random"
# inpath = f"structured_synthetic_generation/simulate/out/{phylo}_lvl_{taxonomic_level}/debug/"
# inpath = f"structured_synthetic_generation/simulate/out/256@rank26_lvl_256@rank26/debug/"


# mask_file = f"structured_synthetic_generation/assemblages/binary_out/100_rich55.1_var10.9/x0_0.csv"
# mask_file = f"structured_synthetic_generation/assemblages/binary_out/256_rich71.8_var17.9/x0_0.csv"

data_file = f'{inpath}data_0.csv'
normed_file = f'{inpath}normed_0.csv'
fitness_file = f'{inpath}fitness_0.csv'



def prepare_data_for_ridgeplot(data_filename, mask_filename):
    df = pd.read_csv(data_filename, header=0)
    mask = pd.read_csv(mask_filename, header=None)

    # Extract feature columns (excluding 'sample' and 'time')
    feature_cols = [col for col in df.columns if col not in ['sample', 'time']]
    
    # Ensure mask shape matches (only use as many rows as there are unique samples)
    num_samples = df['sample'].nunique()
    print(f"Number of samples: {num_samples}")
    mask = mask.iloc[:num_samples]
    mask.columns = feature_cols

    # Create a dict mapping sample_id to feature mask
    sample_masks = {sample: mask.iloc[i].values.astype(bool)
                    for i, sample in enumerate(sorted(df['sample'].unique()))}

    # Apply the mask to set filtered values as negative
    def apply_mask(row):
        sample = row['sample']
        mask_row = sample_masks[sample]
        features = row[feature_cols].values
        masked_features = np.where(mask_row, features, -1.0)  # masked values = -1
        return pd.Series(masked_features, index=feature_cols)

    df[feature_cols] = df.apply(apply_mask, axis=1)

    # Melt to long format
    long_df = df.melt(id_vars=['sample', 'time'], value_vars=feature_cols,
                      var_name='feature', value_name='value')

    # Remove all masked values
    long_df = long_df[long_df['value'] >= 0].copy()

    return long_df



def plot_ridge(data, title):
    quant = 0.75
    outly = 1.5
    zero_threshold = 1e-5

    timepoints = sorted(data['time'].unique())
    num_timepoints = len(timepoints)
    fig, axes = plt.subplots(num_timepoints, 1, figsize=(10, 0.75 * num_timepoints), sharex=True)

    if num_timepoints == 1:
        axes = [axes]

    done = False
    for ax, timepoint in zip(axes, timepoints):
        subset = data[data['time'] == timepoint]
        values = subset['value'].values

        if len(subset) <= 1:
            print(f"All samples collapsed to extinction by timepoint {timepoint}")
            done = True

        # Outlier threshold
        q = np.quantile(values, quant)
        if q > 0:
            threshold = outly * q
        else:
            threshold = zero_threshold

        # Separate non-outliers and outliers
        non_outlier_values = values[values <= threshold]
        outliers = values[values > threshold]

        # Plot histogram only for non-outliershist_range = (0, threshold)  # or (values.min(), threshold) if values are not always >= 0
        hist_range = (0, threshold) 
        counts, bins, patches = ax.hist(
            non_outlier_values,
            bins=20,
            density=True,
            alpha=0.3,
            color='gray',
            range=hist_range
        )

        max_height = counts.max()

        # Plot outliers at half max height
        y_outliers = np.full_like(outliers, 0.5 * max_height)
        ax.scatter(outliers, y_outliers, color='red', s=20, marker='x', zorder=10)

        # Clean up axis
        ax.set_yticks([])
        ax.set_ylabel(f"{timepoint:.1f}", rotation=0, labelpad=20, va='center')
        ax.set_xlim(data['value'].min(), data['value'].max())
        ax.axhline(y=0, color='white', lw=2)

        print(f"Time {timepoint} | q{quant*100 // 1}: {q:.5f} | outlier threshold: {threshold:.5f} | median: {np.median(values):.5f}")
        print(f"Outlier values: {outliers}")


        if done:
            break

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(h_pad=-1)
    plt.subplots_adjust(top=0.95)
    plt.show()


def plot_sum_mags_over_time(data_filename, title, logscale):
    df = pd.read_csv(data_filename, header=0)

    # Get only feature columns
    feature_cols = [col for col in df.columns if col not in ['sample', 'time']]

    # Sum across features per sample
    df['total_sum'] = df[feature_cols].abs().sum(axis=1)

    # Average total sum per time point
    sum_by_time = df.groupby('time')['total_sum'].mean().reset_index()

    # Plot
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=sum_by_time, x='time', y='total_sum', marker='o')
    plt.title(title)
    plt.xlabel('Time')
    if logscale:
        plt.ylabel('Average L1 Magnitude (logscale)')
        plt.yscale('log')
    else:
        plt.ylabel('Average L1 Magnitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# Process each file and create ridge plots
ridge_files = [normed_file]#, data_file, fitness_file]
titles = ['OTU Abundance Distributions vs Time']
# titles = ['Ridge Plot of Nonzero Feature Values Over Time - Abs Abundance', 
#           'Ridge Plot of Nonzero Feature Values Over Time - Growth Rate']

for file, title in zip(ridge_files, titles):
    if os.path.exists(file) and os.path.exists(mask_file):
        df_long = prepare_data_for_ridgeplot(file, mask_file)
        plot_ridge(df_long, title)
    else:
        print(f"File not found: {file} or {mask_file}")

plot_sum_mags_over_time(data_file, 'Absolute Abundance L1 vs Time', logscale=False)

plot_sum_mags_over_time(fitness_file, 'Growth Rate L1 vs Time', logscale=True)


