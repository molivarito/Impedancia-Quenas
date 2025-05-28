import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np # Import numpy for argmax
import math # Import math module for log2

# --- Initial Configuration ---
# Define the directory where your CSV files are located.
# Make sure this path is correct for your system!
data_dir = 'MEDICIONES/Knot'

# Define the directory where the generated plots will be saved.
output_dir = 'knot_images'
# Create the output directory if it doesn't exist.
os.makedirs(output_dir, exist_ok=True)

# Get a list of all .csv files in the specified directory.
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
num_files = len(csv_files)

# List to store all relevant data for the combined summary plot.
summary_data = []

# --- Data Processing and Visualization ---
# Iterate over each CSV file found in the directory.
for idx, file in enumerate(sorted(csv_files)):
    filepath = os.path.join(data_dir, file) # Construct the full path to the current file

    # Create a new figure and an axis for each CSV file.
    # This allows saving each plot independently.
    fig_file, ax_file = plt.subplots(figsize=(10, 6))

    try:
        # Load data from the CSV file using pandas.
        # `sep=';'` specifies that columns are separated by semicolons.
        # `decimal=','` indicates that the comma is the decimal separator (e.g., "1,23" instead of "1.23").
        # `header=None` tells pandas that the file does not have a header row.
        df = pd.read_csv(filepath, sep=';', decimal=',', header=None)

        # Extract the first column as Frequency (x) and the second as Impedance (z).
        x = df.iloc[:, 0]      # Frequency in Hz (first column)
        z = df.iloc[:, 1]      # Impedance (second column)

        # Calculate Admittance (Y), which is the inverse of Impedance (Y = 1/Z).
        # Robust handling for potential division by zero:
        # `.replace(0, pd.NA)`: Replaces any impedance value that is exactly 0 with `pd.NA` (Not Available).
        # `.dropna()`: Removes rows where the resulting admittance would be `NA` (i.e., where the original impedance was 0).
        y = 1.0 / z.replace(0, pd.NA).dropna()

        # After removing rows with zero impedance, `y` indices might not be contiguous.
        # This line ensures that the frequency Series `x` aligns correctly with `y`,
        # keeping only the frequencies corresponding to valid admittance values.
        x = x[y.index]

        # --- Extreme Admittance Filtering ---
        # Filters out extremely large admittance values (close to infinity).
        # These values are often artifacts of impedances very close to zero and
        # distort the plot scale. A threshold of 1e9 is used.
        # Adjust this threshold if your data legitimately has very high admittance values.
        y = y.mask(y > 1e9).dropna()
        x = x[y.index] # Re-align x after removing extreme y values

        # --- Frequency Filtering ---
        # Create a boolean mask to select only frequencies within the desired range:
        # between 200 Hz and 5000 Hz (5 kHz), inclusive.
        mask = (x >= 200) & (x <= 5000)

        # Apply the mask to get the filtered frequency and admittance data.
        # `reset_index(drop=True)` is crucial here; it resets the indices of the filtered Series
        # to be contiguous (0, 1, 2, ...). This is important for `find_peaks` to work correctly,
        # as it expects sequential indices.
        x_filt = x[mask].reset_index(drop=True)
        y_filt = y[mask].reset_index(drop=True)

        # --- Peak Detection ---
        # Use the `find_peaks` function from the SciPy library to identify peaks.
        # ADJUST THESE PARAMETERS (`prominence` and `distance`) TO IMPROVE PEAK DETECTION.
        # `prominence`: Controls the minimum "height" a peak must have to be detected.
        #    - If peaks that SHOULD be seen are NOT detected: REDUCE this value (e.g., 0.01, 0.005).
        #    - If TOO MANY peaks (noise) are detected: INCREASE this value (e.g., 0.1, 0.2).
        # `distance`: Controls the minimum horizontal distance between detected peaks.
        #    - If multiple peaks are detected TOO CLOSE together that should be one: INCREASE this value (e.g., 50, 100).
        #    - If peaks that ARE close are NOT detected: REDUCE this value (e.g., 10, 5).
        peaks, properties = find_peaks(y_filt, prominence=0.04, distance=75)

        # --- Limit to first 3 peaks for visualization and printing ---
        display_peaks_for_plot = peaks[:3] if len(peaks) > 3 else peaks

        # --- Plotting Data on the Current Subplot ---
        # Plot the complete admittance (all data from the file, after extreme value filtering).
        # This provides a visual context of the full curve.
        ax_file.plot(x, y, label='Admittance (Full Data)', color='blue', alpha=0.7)

        if len(display_peaks_for_plot) > 0:
            # Draw a red 'x' marker at the exact location of each detected peak
            # in the filtered data. `markersize` controls the marker size.
            ax_file.plot(x_filt[display_peaks_for_plot], y_filt[display_peaks_for_plot], 'x', color='red', markersize=8, label='Detected Peaks (by freq)')
            # Draw dashed vertical lines from the minimum of the filtered admittance
            # to the value of each peak. This helps visualize the peak "height".
            # Ensure ymin is a positive value for the logarithmic scale.
            min_y_for_vlines = y_filt.min() if y_filt.min() > 0 else 1e-9 # Small positive value for log scale
            ax_file.vlines(x_filt[display_peaks_for_plot], ymin=min_y_for_vlines, ymax=y_filt[display_peaks_for_plot],
                           color='red', linestyle='--', alpha=0.5) # Removed redundant label

        # Configure the title and axis labels for the current plot.
        ax_file.set_title(f'Admittance and Peaks for {file}')
        ax_file.set_xlabel('Frequency (Hz)')
        ax_file.set_ylabel('Admittance (1/Ohm)')
        ax_file.legend()
        ax_file.grid(True, linestyle='--', alpha=0.6)

        # --- Y-axis Adjustment for Better Visualization (Logarithmic Scale) ---
        # Set the Y-axis to a logarithmic scale.
        ax_file.set_yscale('log')
        if not y_filt.empty:
            min_y_log_lim = y_filt[y_filt > 0].min() if not y_filt[y_filt > 0].empty else 1e-12
            ax_file.set_ylim(bottom=min_y_log_lim, top=y_filt.max() * 1.5)
        else:
            ax_file.set_ylim(bottom=1e-12, top=1.0)

        # --- Console Output and Inharmonicity Calculation ---
        print(f"\n--- File: {file} ---")
        if len(peaks) > 0:
            # Print info about the first few detected peaks (by frequency order)
            print("Detected peaks (by frequency) in the 200Hz - 5kHz range:")
            for i, peak_original_idx in enumerate(peaks):
                if i >= 3: # Limit printing to first 3 for brevity
                    if len(peaks) > 3: print(f"  ...and {len(peaks)-3} more peaks.")
                    break
                print(f"  Freq Order Peak {i+1}: {x_filt.iloc[peak_original_idx]:.2f} Hz (Admittance: {y_filt.iloc[peak_original_idx]:.2e})")

            # Select f0 (fundamental)
            f0_index_in_peaks_array = 0 # First peak by frequency is f0
            f0 = x_filt.iloc[peaks[f0_index_in_peaks_array]]
            print(f"  Using f0 (Fundamental): {f0:.2f} Hz (Admittance: {y_filt.iloc[peaks[f0_index_in_peaks_array]]:.2e})")

            f1 = np.nan
            inharmonicity = np.nan
            diameter = None

            # Select f1 (first overtone/harmonic) - highest admittance peak after f0
            if len(peaks) > 1: # Need at least one more peak after f0
                candidate_f1_indices_in_peaks_array = np.arange(1, len(peaks)) # Indices for 'peaks' array, from 1 onwards
                
                # Correctly get admittances of candidate peaks directly from y_filt
                # peaks[candidate_f1_indices_in_peaks_array] gives the actual indices in y_filt for these candidates
                admittances_of_candidates = y_filt.iloc[peaks[candidate_f1_indices_in_peaks_array]]

                if len(admittances_of_candidates) > 0:
                    idx_of_max_in_candidates = np.argmax(admittances_of_candidates) # Index relative to the 'candidates' slice
                    f1_index_in_peaks_array = candidate_f1_indices_in_peaks_array[idx_of_max_in_candidates] # Index in full 'peaks' array
                    f1 = x_filt.iloc[peaks[f1_index_in_peaks_array]]
                    print(f"  Selected f1 (Highest Admittance Peak after f0): {f1:.2f} Hz (Admittance: {y_filt.iloc[peaks[f1_index_in_peaks_array]]:.2e})")

                    # Extract knot diameter from the file name
                    file_name_without_ext = file.replace('.csv', '')
                    diameter_str = file_name_without_ext.replace('mm', '')
                    try:
                        diameter = float(diameter_str.replace(',', '.'))
                    except ValueError:
                        print("  Could not parse diameter from filename.")
                        diameter = None

                    if f0 > 0 and diameter is not None and not np.isnan(f1):
                        inharmonicity = 1200 * math.log2(f1 / (2 * f0))
                        print(f"  Inharmonicity (cents): {inharmonicity:.4f}")
                    else:
                        print("  Cannot calculate inharmonicity (f0 <=0, invalid diameter, or f1 not found).")
                else:
                    print("  No candidate peaks found for f1 after f0.")
            else:
                print("  Only f0 found, not enough peaks to select f1 or calculate inharmonicity.")

            # Store data for the combined summary plot if valid
            if diameter is not None and not np.isnan(f0) and not np.isnan(f1) and not np.isnan(inharmonicity):
                summary_data.append({
                    'diameter': diameter,
                    'fundamental_freq': f0,
                    'harmonic_freq': f1, # This is the selected f1
                    'inharmonicity': inharmonicity,
                    'file_name': file.replace('.csv', '')
                })
            else:
                print(f"  Data for {file} not added to summary due to missing values.")

        else:
            print("No peaks found in the specified frequency range (200Hz - 5kHz).")
        print("-" * (len(file) + 16))

        # Save the individual admittance plot.
        fig_file.tight_layout()
        fig_file.savefig(os.path.join(output_dir, f'{file.replace(".csv", "")}_admitancia.png'))
        plt.close(fig_file) # Close the figure to free up memory

    except Exception as e:
        print(f"Error processing file '{file}': {e}")
        ax_file.set_title(f'Error loading/processing {file}')
        ax_file.text(0.5, 0.5, 'Could not load data', horizontalalignment='center',
                      verticalalignment='center', transform=ax_file.transAxes, color='red')
        fig_file.tight_layout()
        fig_file.savefig(os.path.join(output_dir, f'{file.replace(".csv", "")}_error.png'))
        plt.close(fig_file)


# --- Inharmonicity Plot (Bar Chart, optionally can be removed if combined plot is sufficient) ---
# This bar chart of inharmonicity is kept in case the user prefers it,
# but the new summary plot will also include inharmonicity.
# The summary_data list will be used to generate data for this plot.
if summary_data:
    # Filter only data that has calculated inharmonicity
    inharmonicity_for_bar_chart = [{'file': d['file_name'], 'inharmonicity': d['inharmonicity']}
                                   for d in summary_data if 'inharmonicity' in d]

    if inharmonicity_for_bar_chart:
        files_inharm = [d['file'] for d in inharmonicity_for_bar_chart]
        inharmonicities = [d['inharmonicity'] for d in inharmonicity_for_bar_chart]

        fig_inharm, ax_inharm = plt.subplots(figsize=(12, 7))
        ax_inharm.bar(files_inharm, inharmonicities, color='skyblue')
        ax_inharm.set_xlabel('Measurement File')
        ax_inharm.set_ylabel('Inharmonicity (cents)') # Updated to 'cents'
        ax_inharm.set_title('Inharmonicity between First and Second Peak per File')
        ax_inharm.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right') # Rotate X-axis labels for better readability
        fig_inharm.tight_layout()
        fig_inharm.savefig(os.path.join(output_dir, 'inharmonicity_picos.png'))
        plt.close(fig_inharm) # Close the inharmonicity figure after saving
    else:
        print("\nNo inharmonicity data could be calculated for any file for the bar chart.")
else:
    print("\nNo inharmonicity data could be calculated for any file.")


# --- New Combined Summary Plot: Frequencies and Inharmonicity vs Knot Diameter ---
if summary_data:
    summary_df = pd.DataFrame(summary_data)
    # Ensure diameters are sorted for a consistent line plot
    summary_df = summary_df.sort_values(by='diameter').reset_index(drop=True)

    fig_summary, ax_summary1 = plt.subplots(figsize=(12, 7))

    # Primary Y-axis for Frequencies
    ax_summary1.set_xlabel('Knot Diameter (mm)')
    ax_summary1.set_ylabel('Frequency (Hz)', color='green')
    ax_summary1.tick_params(axis='y', labelcolor='green')

    # Plot fundamental frequencies (first peaks)
    ax_summary1.plot(summary_df['diameter'], summary_df['fundamental_freq'], marker='o', linestyle='-', color='green', linewidth=2, label='Fundamental Frequency (First Peak)')
    # Plot first harmonics (second peaks)
    ax_summary1.plot(summary_df['diameter'], summary_df['harmonic_freq'], marker='x', linestyle='--', color='purple', linewidth=2, label='First Harmonic (Second Peak)')

    # Create a second Y-axis for Inharmonicity
    ax_summary2 = ax_summary1.twinx()
    ax_summary2.set_ylabel('Inharmonicity (cents)', color='red') # Updated to 'cents'
    ax_summary2.tick_params(axis='y', labelcolor='red')

    # Plot inharmonicity
    inharmonicity_line, = ax_summary2.plot(summary_df['diameter'], summary_df['inharmonicity'], marker='s', linestyle=':', color='red', linewidth=2, label='Inharmonicity (cents)') # Updated to 'cents'

    # Add data labels for inharmonicity
    for i, txt in enumerate(summary_df['inharmonicity']):
        ax_summary2.annotate(f'{txt:.2f}', (summary_df['diameter'].iloc[i], summary_df['inharmonicity'].iloc[i]),
                             textcoords="offset points", xytext=(0,10), ha='center', color='red', fontsize=9)

    # Title and combined legend
    fig_summary.suptitle('Summary of Frequencies and Inharmonicity vs Knot Diameter')
    # Combine legends from both axes
    lines1, labels1 = ax_summary1.get_legend_handles_labels()
    lines2, labels2 = ax_summary2.get_legend_handles_labels()
    ax_summary2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax_summary1.grid(True, linestyle='--', alpha=0.7)
    # No need to rotate X-axis labels if they are numerical (diameters)
    # plt.xticks(rotation=45, ha='right')

    # Adjust Y-axis limits for the secondary axis to accommodate labels
    if not summary_df['inharmonicity'].empty:
        min_inharm = summary_df['inharmonicity'].min()
        max_inharm = summary_df['inharmonicity'].max()
        # Give a bit of extra margin at the top for labels
        ax_summary2.set_ylim(bottom=min_inharm * 0.9, top=max_inharm * 1.2)


    fig_summary.tight_layout()
    fig_summary.savefig(os.path.join(output_dir, 'resumen_frecuencias_e_inharmonicidad.png'))
    plt.show() # Show this new summary plot
else:
    print("\nNot enough data to generate the summary plot of frequencies and inharmonicity.")
