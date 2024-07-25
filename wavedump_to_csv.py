import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from multiprocessing import Pool
import scipy.stats as stats


import time

# Record the start time
start_time = time.time()

DATA_DIR = os.getenv("DATA_DIR") + "set2/27/"
print(DATA_DIR)

TRIGGER_FIT_THRESHOLD = 0.35

# Functions
def read_file (filename):
    events = {}

    with open(filename, 'r') as file:
        while True:
            # Read headers
            header_lines = [file.readline().strip() for _ in range(8)]
            if not header_lines[0]:
                break

            # Extract the header information
            record_length = int(header_lines[0].split(":")[1])
            board_id = int(header_lines[1].split(":")[1])
            event_number = int(header_lines[3].split(":")[1])
            if event_number == 0: # Because for some reason it changes after 1st event.
                channel = str(str(header_lines[2].split(":")[1].strip()))
                print(f"Reading channel: {channel}")
            else:
                channel = channel
            pattern = int(header_lines[4].split(":")[1], 16)
            trigger_time_stamp = int(header_lines[5].split(":")[1])
            dc_offset = int(header_lines[6].split(":")[1], 16)
            start_index_cell = int(header_lines[7].split(":")[1])

            # Read the sample points
            samples = []
            for _ in range(record_length):
                sample = file.readline().strip()
                if sample:
                    samples.append(float(sample))
            samples = np.array(samples)

            # process
            bg = np.mean(samples[:30])
            samples = samples-bg
            max_value = np.max(samples)
            dx = 200e-12 #spacing of the data points
            integral = np.trapz(samples[samples > 0], dx=dx)

            event = {
                #'RecordLength': record_length,
                #'BoardID': board_id,
                #'Channel': channel,
                #'EventNumber': event_number,
                #'Pattern': pattern,
                #'TriggerTimeStamp': trigger_time_stamp,
                #'DCOffset': dc_offset,
                #'StartIndexCell': start_index_cell,
                f'Samples.Ch{channel}': np.array(samples),
                f'Background.Ch{channel}': bg,
                f'Max_value.Ch{channel}' : max_value,
                f'Integral.Ch{channel}' : integral,
            }
                                
            events[event_number] = event
    df = pd.DataFrame.from_dict(events, orient='index')# Transpose the DataFrame so that each row is an event

    return df

# Template fitting functions
def landau_func(x, loc, scale, amp):
    z = (x - loc) / scale
    return amp * np.exp(-0.5 * (z + np.exp(-z))) / scale

def trigger_func(x, a, b, c, k):
    return a / (1 + np.exp(-k * (x - c))) + b

def find_plateau_start(waveform, threshold=0.01):
    peak_index = np.argmax(waveform)
    for i in range(peak_index, 0, -1):
        if waveform[i] < waveform[peak_index] * (1 - threshold):
            return i + 1
    return 0

# main analyis function
def analyze_waveform(waveform, channel):
    waveform = np.array(waveform)
    x = np.arange(len(waveform))

    if channel == 'TR_0_0':
        x_range = (20, 220) 
    else:
        x_range = (50, 200)
    
    mask = (x >= x_range[0]) & (x <= x_range[1])
    x_masked = x[mask]
    data_masked = waveform[mask]

    if channel == 'TR_0_0':
        plateau_start = find_plateau_start(waveform)
        fit_range = slice(0, plateau_start + 1)
        x_masked = x[fit_range]
        data_masked = waveform[fit_range]

        rise_start = np.argmax(data_masked > np.min(data_masked) + 0.1 * (np.max(data_masked) - np.min(data_masked)))
        p0 = [np.max(data_masked) - np.min(data_masked),
              np.min(data_masked),
              rise_start,
              1]
        bounds = ([0, -np.inf, 0, 0], 
                  [np.inf, np.inf, len(x_masked), 100])
        
        try:
            popt, _ = curve_fit(trigger_func, x_masked, data_masked, p0=p0, bounds=bounds, maxfev=10000)
            y_fit = trigger_func(x[:plateau_start + 1], *popt)
            
            peak_time = np.argmax(y_fit)
            cfd_level = np.max(y_fit) * 0.38 
            cfd_time_simple = np.interp(cfd_level, y_fit, x[:plateau_start + 1])
        except:
            print(f"Fitting failed for channel {channel} - returning None")
            return pd.Series({'peak_time': None, 'cfd_time_simple': None})

    else:
        p0 = [x_masked[np.argmax(data_masked)], (x_masked[-1] - x_masked[0]) / 10, np.max(data_masked)]
        bounds = ([x_masked[0], 0, 0], [x_masked[-1], x_masked[-1] - x_masked[0], np.inf])

        try:
            popt, _ = curve_fit(landau_func, x_masked, data_masked, p0=p0, bounds=bounds, maxfev=10000)
            y_fit = landau_func(x, *popt)

            peak_time = popt[0]
            cfd_level = y_fit.max() * 0.38
            cfd_index = np.argmax(y_fit > cfd_level)
            cfd_time_simple = np.interp(cfd_level, [y_fit[cfd_index-1], y_fit[cfd_index]], [x[cfd_index-1], x[cfd_index]])
        except:
            print(f"Fitting failed for channel {channel} - returning None")
            return pd.Series({'peak_time': None, 'cfd_time_simple': None})

    return pd.Series({
        'peak_time': peak_time,
        'cfd_time_simple': cfd_time_simple,
    })

# for multithreading the analysis
def process_single_waveform(args):
    waveform, channel = args
    return analyze_waveform(waveform, channel)

def parallel_process_waveforms(df, channels, num_processes=8):
    all_results = []
    
    for channel in channels:
        column_name = f'Samples.Ch{channel}'
        if column_name not in df.columns:
            continue
        
        waveforms = df[column_name].tolist()
        
        with Pool(num_processes) as pool:
            channel_results = pool.map(process_single_waveform, [(waveform, channel) for waveform in waveforms])
        
        channel_df = pd.DataFrame(channel_results)
        channel_df.columns = [f'{col}.Ch{channel}' for col in channel_df.columns]
        all_results.append(channel_df)
    
    return pd.concat(all_results, axis=1)


def main():
    # define channels to read in
    channels = [0,1]
    files = ["TR_0_0.txt"] + [f"wave_{channel}.txt" for channel in channels] # ensures trigger is being read in

    df_tr = read_file(DATA_DIR + files[0])
    df_0 = read_file(DATA_DIR + files[1])
    df_1 = read_file(DATA_DIR + files[2])

    dfs = df_tr, df_0, df_1
    df_merged = pd.concat(dfs, axis=1)
    print("Finished reading files.")

    timing_channels = ['TR_0_0', '0'] 
    num_processes = 8  # Adjust based on your CPU cores

    # Process all waveforms in parallel
    print("Fitting and processing waves...")
    results = parallel_process_waveforms(df_merged, timing_channels, num_processes)
    # Add the results to your original DataFrame
    results_df = pd.concat([df_merged, results], axis=1)
    print("Finishing fitting.")

    columns_to_drop = [col for col in results_df.columns if "Samples" in col or "Background" in col]
    columns_to_drop.extend(['peak_time.ChTR_0_0', 'cfd_time_simple.Ch0'])

    results_df = results_df.drop(columns=columns_to_drop)
    results_df = results_df.rename(columns=
                                   {'cfd_time_simple.ChTR_0_0': 'trigger_time'})

    # Calculate the time difference
    results_df['peak_time.Ch0'] = results_df['peak_time.Ch0']*200
    results_df['trigger_time'] = results_df['trigger_time']*200
    time_diff = (results_df['peak_time.Ch0']- results_df['trigger_time'])
    results_df['ch0-trigger_time'] = time_diff

    # Calculate mean and standard deviation
    mean_diff = np.mean(time_diff)
    std_diff = np.std(time_diff)

    
    # Plot the histogram
    plt.hist(time_diff, bins=100, density=True, alpha=0.6, color='g')

    # Fit a Gaussian distribution to the data
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean_diff, std_diff)
    plt.plot(x, p, 'k', linewidth=2)

    # Set axis labels
    plt.xlabel('Time Difference (ps)')
    plt.ylabel('Frequency')

    # Add grid
    plt.grid(True)

    # Annotate mean and std on the plot
    textstr = f'Mean: {mean_diff:.2f} ps\nStd: {std_diff:.2f} ps'
    plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='white'))

    # Show plot
    plt.show()

    # Record the end time
    end_time = time.time()

    # Calculate and print the runtime
    runtime = end_time - start_time
    print(f"Runtime: {runtime:.2f} seconds")

    # Get the output directory from the environment variable and save
    out_dir = os.getenv('OUT_DIR', '.')
    output_file = os.path.join(out_dir, 'SiPM.csv')
    results_df.to_csv(output_file, index=False)
    print(f"Finished processing. DataFrame has been saved to {output_file}")


if __name__ == "__main__":
    main()