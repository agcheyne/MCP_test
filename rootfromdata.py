import pandas as pd
import numpy as np
import uproot
import awkward as ak
from tqdm import tqdm
import os


def read_detector_data(file_path):
    print(f"Reading file: {file_path}")
    events = []
    current_event = None
    values = []

    # Count total lines for progress bar
    with open(file_path, 'r') as file:
        total_lines = sum(1 for _ in file)
        file.seek(0)  # Reset file pointer to beginning

        with tqdm(total=total_lines, desc=f"Reading {file_path}", unit="lines") as pbar:
            for line in file:
                pbar.update(1)
                line = line.strip()
                if not line:
                    continue
                if line.startswith('Record Length:'):
                    if current_event:
                        current_event['values'] = np.array(values)
                        events.append(current_event)
                    current_event = {'Record Length': int(line.split(':')[1])}
                    values = []
                elif line.startswith('Event Number:'):
                    current_event['Event Number'] = int(line.split(':')[1])
                elif line.startswith('Trigger Time Stamp:'):
                    current_event['Trigger Time Stamp'] = int(line.split(':')[1])
                elif line[0].isdigit():
                    values.append(float(line))

    # Add the last event
    if current_event:
        current_event['values'] = np.array(values)
        events.append(current_event)

    # Create DataFrame
    df = pd.DataFrame(events)
    print(f"Finished reading {len(df)} events from {file_path}")
    return df

def process_detector_data(df, channel):
    print(f"Processing data for channel {channel}")
    threshold = 0.5  # 50% of max value

    df['bg'] = df['values'].apply(lambda x: x[:40].mean())
    df['values'] = df['values'].apply(lambda x: x - np.mean(x[:40]))
    df['max'] = df['values'].apply(lambda x: x.max())
    df['max_index'] = df['values'].apply(lambda x: x.argmax())
    df['leading_edge'] = df['values'].apply(lambda x: np.argmax(x > x.max() * threshold))

    if channel != 99:
        #df['integral'] = df['values'].apply(lambda x: x[x > 0].sum())
        df['integral'] = df['values'].apply(lambda x: np.trapz(x[x > 0]))
        df['trailing_edge'] = df['values'].apply(lambda x: len(x) - 1 - np.argmax(x[::-1] > x.max() * threshold))
        df['tot'] = df['trailing_edge'] - df['leading_edge']
        # Add blank time diff column for this channel
        df['time_diff'] = 0

    df.drop(columns=['values', 'bg', 'Record Length'], inplace=True)
    df.columns = [str(channel) + '_' + col if col not in ['Event Number', 'Trigger Time Stamp'] else col for col in df.columns]
    print(f"Finished processing data for channel {channel}")
    return df

def main():
    base_path = '/home/agcheyne/Code/EIC/MCPtest/data/set2/27'

    # Waveform data
    channels = [99, 0, 1]  # List of channels
    
    dfs = []  # List to store DataFrames for each channel
    
    for channel in channels:
        if channel == 99:
            filename = f'TR_0_0.txt'
        else:
            filename = f'wave_{channel}.txt'
        df = read_detector_data(f'{base_path}/{filename}')
        df = process_detector_data(df, channel)
        dfs.append(df)  # Add processed DataFrame to the list

    # Merge all channel DataFrames
    merged_df = pd.concat(dfs, axis=1)

    print(merged_df.head())

    #remove duplicate columns
    merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]
    print(merged_df.head())

    #sort alphabetically
    merged_df = merged_df.reindex(sorted(merged_df.columns), axis=1)
    print(merged_df.head())

    # update time_diff column
    for channel in channels:
        if channel != 99:
            merged_df[str(channel) + '_time_diff'] = merged_df[str(channel) + '_leading_edge'] - merged_df['99_leading_edge']

    #drop unnecessary columns here if needed
         

    print("Converting to Awkward Array")
    ak_array = ak.Array(merged_df.to_dict(orient='list'))

    output_file = "27.root"
    print(f"Saving to ROOT file: {output_file}")
    
   # Remove the existing file if it exists
    if os.path.exists(output_file):
       os.remove(output_file)
       print(f"Removed existing file: {output_file}")

    # Create the new ROOT file
    with uproot.recreate(output_file) as f:
       f["tree"] = ak_array

    print(f"DataFrame saved as a ROOT TTree in '{output_file}'")

if __name__ == "__main__":
    main()
