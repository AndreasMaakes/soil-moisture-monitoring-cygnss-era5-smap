import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CYGNSS.import_data import importData

def bin_and_plot_snr_with_curve(
    folder_name,
    bin_width=5,
    min_angle=0.0,
    max_angle=90.0,
    poly_deg=3,
    figsize=(8,5)
):
    # 1) load & concatenate
    dfs = importData(folder_name)
    df  = pd.concat(dfs, ignore_index=True)

    # 2) fixed-width bins
    edges = np.arange(min_angle, max_angle + bin_width, bin_width)
    df['inc_bin'] = pd.cut(df['sp_inc_angle'], bins=edges, right=False)

    # 3) compute bin means & counts
    stats = ( df.groupby('inc_bin')['ddm_snr']
                .agg(['mean','count'])
                .reset_index() )
    stats['angle_mid'] = stats['inc_bin'].apply(lambda iv: iv.left + bin_width/2)

    # 4) drop bins with too few samples
    stats = stats[stats['count'] > 10]

    # 5) scatter the binned means
    plt.figure(figsize=figsize)
    plt.scatter(stats['angle_mid'],
                stats['mean'],
                s=60,
                color='C0',
                label='binned mean')

    # 6) fit & plot a smooth polynomial curve
    x = stats['angle_mid'].values
    y = stats['mean'].values
    coeffs = np.polyfit(x, y, deg=poly_deg)
    x_smooth = np.linspace(min_angle, max_angle, 300)
    y_smooth = np.polyval(coeffs, x_smooth)
    plt.plot(x_smooth,
             y_smooth,
             color='k',
             linestyle='--',
             linewidth=1.5,
             label=f'poly deg={poly_deg}')

    # 7) finalize
    plt.xlabel('Incidence angle (°)')
    plt.ylabel('Mean DDM SNR (dB)')
    plt.title(f'Mean DDM SNR vs. Incidence Angle ({bin_width}° bins)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()



bin_and_plot_snr_with_curve(
        "Uruguay/Uruguay-20200101-20200107",
        bin_width=2,
        poly_deg=2
    )
