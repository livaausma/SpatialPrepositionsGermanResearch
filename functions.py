import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import numpy as np
from scipy import stats  # Import the scipy stats module
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency


    
def get_values(task_nr):
    """
    Retrieve the values associated with a given task number.

    Parameters:
    task_nr (str): The task number as a string (e.g., "S1", "A21", "X1").

    Returns:
    tuple: A tuple containing the following elements:
        - int: The column index or starting point for the task.
        - str: The color associated with the task.
        - int: The number of columns or some other integer value related to the task.
    """
    # Dictionary mapping task numbers to their respective values
    task_map = {
        "S1": (0, "Tan", 1),
        "S2": (1, "CadetBlue", 1),
        "S3": (2, "SlateBlue", 1),
        "S4": (3, "IndianRed", 1),
        "S5": (4, "DarkGoldenRod", 1),
        "S6": (5, "PaleVioletRed", 1),
        "S7": (6, "SteelBlue", 1),
        "S8": (7, "Teal", 1),
        "S9": (8, "RosyBrown", 1),
        "A21": (9, "Gold", 4),
        "A22": (13, "Sienna", 4),
        "A23": (17, "Darkkhaki", 4),
        "A24": (21, "Goldenrod", 4),
        "A25": (25, "Olive", 4),
        "A26": (29, "Chocolate", 4),
        "A31": (33, "Khaki", 9),
        "A32": (42, "Skyblue", 9),
        "A33": (51, "MediumAquamarine", 9),
        "A34": (60, "PaleVioletRed", 9),
        "A35": (69, "DarkSalmon", 9),
        "A36": (78, "MediumPurple", 9),
        "X1": (87, "darkslateblue", 2),
        "X2": (89, "lightskyblue", 2),
        "X3": (91, "slategray", 2),
        "X4": (93, "rebeccapurple", 2),
        "X5": (95, "orchid", 2),
        "X6": (97, "mediumslateblue", 2),
        "X7": (99, "cornflowerblue", 2),
        "X8": (101, "darkorchid", 2),
        "X9": (103, "dodgerblue", 2)
    }
    return task_map.get(task_nr, "Error: Invalid task number")


# Plots results of one dataset from the SCT
def plot_completion_one(abbreviation, number, colorname, csv_file_path, plot_save_path, save_dpi, save, show, encoding='ISO-8859-1'):      
    """
    Creates a bar chart of word frequencies from a specified CSV column and computes statistics.

    This function reads a CSV file, processes the specified column to count occurrences
    of 'auf' and 'über', and plots their frequencies in a bar chart. It calculates the 
    mean and variance of the categorized values, displaying this information on the plot.

    Parameters:
    abbreviation (str): Abbreviation used in the plot title.
    number (int): Index of the column to process in the CSV file.
    colorname (str): Color of the bars in the plot.
    csv_file_path (str): Path to the CSV file.
    plot_save_path (str): Path where the plot will be saved.
    save_dpi (int): DPI for the saved plot.
    save (bool): Indicates whether to save the plot.
    show (bool): Indicates whether to display the plot.
    encoding (str, optional): File encoding, default is 'ISO-8859-1'.

    Returns:
    None
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path, encoding=encoding)
        column = df.iloc[:, number]  
        
        # Extract the values from the first two rows of the column
        first_row = column.iloc[0] if not column.empty else "Unknown"
        second_row = column.iloc[1] if len(column) > 1 else "Unknown"
        
        # Create a title using the first abbreviation and first two row values
        title = abbreviation.upper() + ": " + f'{first_row}\n{second_row}'


        # Exclude the first two rows from the value counts
        filtered_data = column.iloc[2:].dropna()  # Skip the first two rows and handle missing data

        # Categorize the data
        def categorize_word(word):
            if word == 'auf':
                return 1
            elif word == 'über':
                return 2
            else:
                return None  # Ignore any word that is not "auf" or "über"

        # Apply the categorization
        categorized_data = filtered_data.apply(categorize_word)

        # Filter out None values (i.e., other words)
        categorized_data = categorized_data.dropna()

        # Calculate the mean and variance only for "auf" and "über"
        mean_value = categorized_data.mean()
        variance_value = categorized_data.var()

        # Get the value counts for only "auf" and "über"
        value_counts = categorized_data.value_counts().sort_index()  # Sort by index to ensure order

        # Set up a smaller plot size
        plt.figure(figsize=(6, 6))  # Adjust the width and height to be smaller

        # Plot the bar chart with customizations
        ax = plt.subplot(1, 1, 1)

        # Plot the bars with a width of 0.5
        ax.bar(value_counts.index.map({1: 'auf', 2: 'über'}), value_counts.values, color=colorname, width=0.5)

        # Set y-axis limits to control bar height if necessary
        max_value = value_counts.max()
        tick_interval = 5
        ax.set_ylim(0, max_value + tick_interval)  # Limit height of the bars

        # Set custom y-axis ticks
        custom_ticks = np.arange(0, max_value + tick_interval + tick_interval, tick_interval)
        ax.set_yticks(custom_ticks)

        # Adding title and labels
        ax.set_title(title, fontsize=14)  # Slightly smaller font size for the title
        ax.set_xlabel('Categories', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)

        # Ensure x-axis labels are not rotated
        ax.tick_params(axis='x', labelrotation=0)

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Add mean and variance information below the plot
        plt.figtext(0.1, -0.1, f'Mean: {mean_value:.2f}, Variance: {variance_value:.2f}', fontsize=14, ha='left')

        # Save the plot
        if save is True:
            plt.savefig(plot_save_path, dpi=save_dpi, bbox_inches='tight')

        # Show the plot
        if show is True:
            plt.show()

    except Exception as e:
        # Print the error message if an exception occurs
        print(f"Error reading CSV file or plotting: {e}")


# Plots results of multiple datasets (comparisons) from SCT
def plot_completion_multiple(abbreviation_list, start_col_list, colorname_list, csv_file_path, column_number, plot_save_path, save_dpi, save, show, encoding='ISO-8859-1'):
    """
    Creates multiple bar plots comparing word frequencies in specified CSV columns.

    This function reads a CSV, processes selected columns to categorize 'auf' and 'über',
    and generates individual plots for each column. If two datasets are compared, 
    a t-test is performed, and the t-statistic and p-value are displayed on the plot.

    Parameters:
    abbreviation_list (list of str): Abbreviations for plot titles.
    start_col_list (list of int): Start indices of columns to process.
    colorname_list (list of str): Colors for each plot.
    csv_file_path (str): Path to the CSV file.
    column_number (int): Number of columns to compare per row.
    encoding (str, optional): CSV encoding, defaults to 'ISO-8859-1'.

    Returns:
    None
    """

    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path, encoding=encoding)

        # Check that the lists of start columns and colors are of the same length
        if len(start_col_list) != len(colorname_list):
            raise ValueError("The length of start_col_list and colorname_list must be the same.")

        # Calculate the number of rows and columns for subplots
        num_rows = len(start_col_list)  # Number of rows is the length of start_col_list
        num_cols = int(column_number)   # Number of columns is the column_number

        # Initialize a plot with a larger figure size to control overall height
        plt.figure(figsize=(num_cols * 5, num_rows * 4))  # Adjust height here

        # Initialize variables to store data for comparison
        datasets = []

        # Loop through each start_col and corresponding color
        for abbreviation, start_col, colorname in zip(abbreviation_list, start_col_list, colorname_list):
            # Collect the averages and labels for each column
            for i in range(int(column_number)):
                column = df.iloc[:, start_col + i]
                
                # Extract the values from the first two rows of the column
                first_row = column.iloc[0] if not column.empty else "Unknown"
                second_row = column.iloc[1] if len(column) > 1 else "Unknown"
                
                # Create a title using the first abbreviation and first two row values
                title = abbreviation.upper() + ": " + f'{first_row}\n{second_row}'

                # Exclude the first two rows from the value counts
                filtered_data = column.iloc[2:]  # Skip the first two rows

                # Function to categorize the words
                def categorize_word(text):
                    if "nicht" in text:
                        return None  # Exclude "nicht auf"
                    if text == 'auf':
                        return 'auf'
                    elif text == 'über':
                        return 'über'
                    elif "auf" in text:
                        return 'other with auf'
                    else:
                        return None  # Ignore words that don't fit the categories

                # Apply the categorization to the filtered data
                categorized_data = filtered_data.apply(categorize_word)

                # Filter out None values (i.e., other words or 'nicht auf')
                categorized_data = categorized_data.dropna()

                # Add dataset to a list for comparison if we have two datasets
                datasets.append(categorized_data)

                # Get the value counts for the three categories
                value_counts = categorized_data.value_counts().sort_index()

                # Plot the bar chart for each column
                ax = plt.subplot(num_rows, num_cols, start_col_list.index(start_col) * int(column_number) + i + 1)

                # Plot the bars with a specific width
                ax.bar(value_counts.index, value_counts.values, color=colorname, width=0.5)  # Adjust width here

                # Set y-axis limits to control bar height if necessary
                max_value = value_counts.max()
                tick_interval = 5
                ax.set_ylim(0, max_value + tick_interval)  # Limit height of the bars

                # Set custom y-axis ticks
                custom_ticks = np.arange(0, max_value + tick_interval + tick_interval, tick_interval)
                ax.set_yticks(custom_ticks)

                # Adding title and labels
                ax.set_title(title, fontsize=14)
                ax.set_xlabel('Categories', fontsize=14)
                ax.set_ylabel('Frequency', fontsize=14)
                
                # Ensure x-axis labels are not rotated
                ax.tick_params(axis='x', labelrotation=0)

                # # Add mean and variance information below the plot
                # mean_value = categorized_data.map({'auf': 1, 'über': 2, 'other with auf': 3}).mean()
                # variance_value = categorized_data.map({'auf': 1, 'über': 2, 'other with auf': 3}).var()
                # ax.text(0.92, 0.6, f'Mean: {mean_value:.2f}\nVariance: {variance_value:.2f}', transform=ax.transAxes, fontsize=14, ha='right')

                # Adjust spacing between subplots (increase height space)
                plt.subplots_adjust(bottom=0.4, hspace=1)  # Adjust hspace to increase the vertical space between plots

        # If we have exactly two datasets, perform the t-test
        if len(datasets) == 2:
            # Mapping the categories 'auf', 'über', 'other with auf' to numeric values
            category_mapping = {'auf': 1, 'other with auf': 2, 'über': 3}
            
            # Convert datasets to numeric form using the mapping
            numeric_data1 = datasets[0].map(category_mapping).dropna()
            numeric_data2 = datasets[1].map(category_mapping).dropna()
            
            # Perform an independent t-test
            t_stat, p_value = ttest_ind(numeric_data1, numeric_data2, equal_var=False)  # Set equal_var=False for Welch's t-test

            # Display the t-test result on the last plot
            plt.figtext(0.5, -0.05, f'T-test statistic: {t_stat:.2f}, p-value: {p_value:.6f}', fontsize=14, ha='center', va='top')

        # If we have more than two datasets, perform the chi-square test
        elif len(datasets) > 2:
            categories = ['auf', 'other with auf', 'über']

            # Convert the datasets to categorical with the predefined categories
            dataset_counts = [pd.Categorical(dataset, categories=categories).value_counts() for dataset in datasets]

            # Combine datasets into a contingency table
            contingency_table = pd.DataFrame(dataset_counts, index=[f'Data{i+1}' for i in range(len(datasets))], columns=categories).T

            # Perform chi-square test
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Display Chi-square test result on the last plot
            plt.figtext(0.5, -0.05, f'Chi-Square statistic: {chi2_stat:.2f}, p-value: {p_value:.6f}', fontsize=14, ha='center', va='top')
        # Adjust layout to prevent overlap
        plt.tight_layout()

         # Save the plot
        if save is True:
            plt.savefig(plot_save_path, dpi=save_dpi, bbox_inches='tight')

        # Show the plot
        if show is True:
            plt.show()
        
    except Exception as e:
        # Print the error message if an exception occurs
        print(f"Error reading CSV file or plotting: {e}")


# Plots for one dataset for first and second SAT
def plot_avg_assessment_one_A(abbreviation, start_col, colorname, csv_file_path, column_number, plot_save_path, save_dpi, save, show, encoding='ISO-8859-1'):
    """
    Creates multiple bar plots comparing word frequencies in specified CSV columns.

    This function reads a CSV file, processes selected columns to categorize occurrences
    of 'auf', 'über', and 'other with auf', and generates individual bar plots for each
    column. If exactly two datasets are compared, a t-test is performed, and the t-statistic 
    and p-value are displayed on the last plot. If more than two datasets are present, a 
    chi-square test is conducted, and its results are also displayed.

    Parameters:
    abbreviation_list (list of str): Abbreviations for the plot titles corresponding to each column.
    start_col_list (list of int): Start indices of the columns to process in the CSV file.
    colorname_list (list of str): Colors for each bar plot.
    csv_file_path (str): Path to the CSV file.
    column_number (int): Number of columns to compare per row of the plots.
    plot_save_path (str): Path where the plots will be saved.
    save_dpi (int): DPI for the saved plots.
    save (bool): Indicates whether to save the plots.
    show (bool): Indicates whether to display the plots.
    encoding (str, optional): File encoding for the CSV, defaults to 'ISO-8859-1'.

    Returns:
    None
    """

    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path, encoding=encoding)
        
        # Get the data from the starting column
        columns = df.iloc[:, start_col]
        first_row = columns.iloc[0] if not columns.empty else "Unknown"
        
        # Initialize list to store averages, variances, and column names
        stats_list = []

        # Loop through the specified columns for mean and variance calculation
        for i in range(column_number):
            # Get the data for the current column
            column = df.iloc[:, start_col + i]
            column_data = column.iloc[2:]  # Skip the first two rows
            
            # Get the column name (from the first row in the column)
            column_name = column.iloc[1] if not column.empty else "Unknown"
            
            # Convert column data to integers
            digits = [int(digit) for digit in column_data]
            
            # Calculate the mean and variance of the digits
            average = sum(digits) / len(digits)
            variance = sum((x - average) ** 2 for x in digits) / len(digits)
            
            # Append the mean, variance, and column name to the list
            stats_list.append([average, variance, column_name])

        # Convert the list to a DataFrame for easier plotting
        stats_df = pd.DataFrame(stats_list, columns=['Average', 'Variance', 'Column'])

        # Extract the second, fourth, and sixth words from the column names and format with newlines
        def create_short_label(label):
            parts = label.split()
            if len(parts) > 5:
                # Combine the necessary parts and remove periods from the 6th part
                return '\n'.join([parts[1], parts[3], parts[5].replace('.', '')])
            else:
                # Return the original label if there are not enough parts
                return label

        def replace_kugelschreiber(short_label):
            if column_number == 9 and "Kugelschreiber" in short_label:
                return short_label.replace("Kugelschreiber", "KS")
            return short_label

        # Apply the functions to create and adjust short labels
        stats_df['Short_Label'] = stats_df['Column'].apply(create_short_label).apply(replace_kugelschreiber)

        # Plot the bar chart for averages
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.bar(stats_df['Short_Label'], stats_df['Average'], color=colorname, alpha=0.6, label='Average')

        # Add the y-axis label for the bar chart
        ax1.set_ylabel('Value', color='black')

        # Rotate x-axis labels if there are 9 bars
        if len(stats_df) == 9:
            ax1.tick_params(axis='x', rotation=90)  # Rotate x-axis labels

        # Create a secondary axis for variance
        ax2 = ax1.twinx()
        ax2.plot(stats_df['Short_Label'], stats_df['Variance'], 'ks', label='Variance')  # Black square markers
        ax2.plot(stats_df['Short_Label'], stats_df['Variance'], 'k--')  # Black dotted line

        # Set the y-axis limits for both axes to [0, 5]
        ax1.set_ylim(0, 5.2)
        ax2.set_ylim(0, 2.8)

        # Add the y-axis label for the variance
        ax2.set_ylabel('Value', color='black')

        # Create a title using the first abbreviation and first row 
        # title = abbreviation.upper() + ": " + f'{first_row}'
        # plt.title(title)

        fig.tight_layout()    
        plt.legend(loc='best')

        # Save the plot
        if save is True:
            plt.savefig(plot_save_path, dpi=save_dpi, bbox_inches='tight')

        # Show the plot
        if show is True:
            plt.show()

    except Exception as e:
        # Print the error message if an exception occurs
        print(f"Error reading CSV file or plotting: {e}")





# Plots for mutliple datasets (comparisons) for first and second SAT
def plot_avg_assessment_multiple_A(abbreviation_list, start_col_list, colorname_list, csv_file_path, column_number, plot_save_path, save_dpi, save, show, encoding='ISO-8859-1'):
    """
    Plots comparison of averages and variances across multiple columns.

    This function reads a CSV file, processes selected columns to calculate the average and variance 
    for each column's values, and creates bar plots. It compares groups of columns using t-tests, 
    displaying the results on the plots. Each plot is color-coded, and column labels are derived from 
    the abbreviation list and column names. Annotations are added for t-test results between groups.

    Parameters:
    abbreviation_list (list of str): Abbreviations for plot titles corresponding to each dataset.
    start_col_list (list of int): Starting column indices to process in the CSV file.
    colorname_list (list of str): Colors for each plot group.
    csv_file_path (str): Path to the CSV file.
    column_number (int): Number of columns to process per group.
    plot_save_path (str): Path where the plots will be saved.
    save_dpi (int): DPI for the saved plots.
    save (bool): Indicates whether to save the plots.
    show (bool): Indicates whether to display the plots.
    encoding (str, optional): Encoding for reading the CSV file, default is 'ISO-8859-1'.

    Returns:
    None
    """
    

    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path, encoding=encoding)

        # Ensure that column_number does not exceed the available columns
        max_col_index = df.shape[1] - 1  # Get the max column index

        # Initialize lists
        combined_avg_var_and_name_list = []
        first_row_list = []
        group_data = []  # List to store data for t-test
        
        # Loop through each start_col and corresponding color
        for abbreviation, start_col, colorname in zip(abbreviation_list, start_col_list, colorname_list):
            avg_var_and_name_list = []
            data_for_ttest = []
            
            # Collect the averages, variances, and labels for each question
            for i in range(int(column_number)):
                col_index = start_col + i
                if col_index > max_col_index:
                    print(f"Column index {col_index} is out of bounds.")
                    continue
                
                column = df.iloc[:, col_index]
                first_row = abbreviation.upper() + ": " + column.iloc[0] if not df.empty else "Unknown"
                column_data = column.iloc[2:]
                column_name = column.iloc[1] if not column.empty else "Unknown"

                # Ensure the column name has enough words
                split_column_name = column_name.split()
                if len(split_column_name) < 6:
                    print(f"Column name does not have enough words: {column_name}")
                    continue

                # Separate the digits
                digits = [int(digit) for digit in column_data]
                # Calculate the mean
                average = sum(digits) / len(digits)
                # Calculate the variance
                variance = sum((x - average) ** 2 for x in digits) / len(digits)

                # Append data for t-test
                data_for_ttest.append(digits)
                
                # Append the average, variance, column name, and color to the list
                avg_var_and_name_list.append([average, variance, column_name, colorname, first_row])

            # Append this question's data to the combined list
            combined_avg_var_and_name_list.append(avg_var_and_name_list)
            first_row_list.append(first_row)
            group_data.append(data_for_ttest)
        
        # Perform t-test between groups (assuming two groups here)
        ttest_results = []
        if len(group_data) == 2:  # Ensure there are exactly two groups for comparison
            group1, group2 = group_data
            for i in range(min(len(group1), len(group2))):  # Compare each question's data
                t_stat, p_val = ttest_ind(group1[i], group2[i], equal_var=False)  # Welch’s t-test
                ttest_results.append((t_stat, p_val))
        
        # Reorder the data for plotting: interleave the groups
        reordered_list = []
        for i in range(int(column_number)):  # Loop through the indices within each group
            for group in combined_avg_var_and_name_list:
                if i < len(group):
                    reordered_list.append(group[i])

        # Convert the reordered list to a DataFrame
        final_df = pd.DataFrame(reordered_list, columns=['Average', 'Variance', 'Column', 'Color', 'First_Row'])

        final_df['Short_Label'] = final_df['Column'].apply(
            lambda x: f"{x.split()[1]} {x.split()[3]} {x.split()[5].replace('.', '')}" 
            if len(x.split()) <= 6 else f"{x.split()[1]} {x.split()[3]} {x.split()[5]} {x.split()[6].replace('.', '')}"
        )
        
        # Plot the bar chart for averages
        fig, ax1 = plt.subplots(figsize=(2*(len(abbreviation_list) + 4), 9))
        ax1.bar(final_df['Short_Label'], final_df['Average'], color=final_df['Color'], alpha=0.6)

        # Create a secondary axis for variance
        ax2 = ax1.twinx()
        ax2.plot(final_df['Short_Label'], final_df['Variance'], 'ks', label='Variance')  # Black square markers
        ax2.plot(final_df['Short_Label'], final_df['Variance'], 'k--', label='Variance Line')  # Black dotted line

        # Set the y-axis limits for both axes
        ax1.set_ylim(0, 5.2)
        ax2.set_ylim(0, 2.8)

        # Add labels and titles
        ax1.set_ylabel('Average Value', color='black')
        ax2.set_ylabel('Variance', color='black')
        # plt.title('Comparison of Averages and Variances')

        # Rotate x-axis labels to make them readable
        ax1.tick_params(axis='x', rotation=90)  # Apply rotation directly to the axis
        
        # Add legends for both Average and Variance
        legend_entries = [
            Line2D([0], [0], color=colorname, lw=4, label=first_row)
            for colorname, first_row in zip(colorname_list, first_row_list)
        ]
        # Add legends to the plot, placing it outside the plot area
        ax1.legend(handles=legend_entries, loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=2) # ends up above title
        # Add legend to the upper right corner in the plot area
        ax2.legend(loc='upper right')
        # Ensure everything is laid out correctly
        plt.tight_layout()

        if len(ttest_results) < 5:
            # Define positions for annotations
            positions = {
                '1-2': (0.205, 0.00),
                '3-4': (0.29, 0.00),
            }
            # Annotate 1-3
            q1_q2_results = ttest_results[:2]
            for i, (t_stat, p_val) in enumerate(q1_q2_results):
                plt.figtext(positions['1-2'][0], positions['1-2'][1] - i * 0.03, 
                            f'{i+1}: t-stat={t_stat:.2f}, p-val={p_val:.3f}', fontsize=14, ha='left')
            # Annotate Q4-Q6
            q3_q4_results = ttest_results[2:4]
            for i, (t_stat, p_val) in enumerate(q3_q4_results):
                plt.figtext(positions['3-4'][0] + 0.33, positions['3-4'][1] - i * 0.03, 
                            f'{i+3}: t-stat={t_stat:.2f}, p-val={p_val:.3f}', fontsize=14, ha='left')
            # Draw vertical lines between x-axis labels
            num_labels = len(final_df['Short_Label'])
            if num_labels >= 4:
                for i in range(len(colorname_list), num_labels, len(colorname_list)):
                    # Draw vertical line in the plot area
                    ax1.axvline(x=i - 0.5, color='gray', linestyle='--', linewidth=0.8)  # Adjust ymax to extend below the x-axis labels


        # Annotate t-test results
        if len(ttest_results) > 5:
            # Define positions for annotations
            positions = {
                '1-3': (0.02, 0.00),
                '4-6': (0.05, 0.00),
                '7-9': (0.08, 0.00)
            }
            # Annotate Q1-Q3
            q1_q3_results = ttest_results[:3]
            for i, (t_stat, p_val) in enumerate(q1_q3_results):
                plt.figtext(positions['1-3'][0], positions['1-3'][1] - i * 0.03, 
                            f'{i+1}: t-stat={t_stat:.2f}, p-val={p_val:.3f}', fontsize=14, ha='left')
            # Annotate Q4-Q6
            q4_q6_results = ttest_results[3:6]
            for i, (t_stat, p_val) in enumerate(q4_q6_results):
                plt.figtext(positions['4-6'][0] + 0.33, positions['4-6'][1] - i * 0.03, 
                            f'{i+4}: t-stat={t_stat:.2f}, p-val={p_val:.3f}', fontsize=14, ha='left')
            # Annotate Q7-Q9 if available
            if len(ttest_results) > 6:
                q7_q9_results = ttest_results[6:9]
                for i, (t_stat, p_val) in enumerate(q7_q9_results):
                    plt.figtext(positions['7-9'][0] + 0.67, positions['7-9'][1] - i * 0.03, 
                                f'{i+7}: t-stat={t_stat:.2f}, p-val={p_val:.3f}', fontsize=14, ha='left')
            # Draw vertical lines between x-axis labels
            num_labels = len(final_df['Short_Label'])
            if num_labels > 6:
                for i in range(len(colorname_list), num_labels, len(colorname_list)):
                    # Draw vertical line in the plot area
                    ax1.axvline(x=i - 0.5, color='gray', linestyle='--', linewidth=0.8)  # Adjust ymax to extend below the x-axis labels
        
        # Save the plot
        if save is True:
            plt.savefig(plot_save_path, dpi=save_dpi, bbox_inches='tight')
        
        # Show the plot
        if show is True:
            plt.show()

    except Exception as e:
        print(f"Error reading CSV file or plotting: {e}")
        


# Plots for one dataset of third SAT
def plot_avg_assessment_one_X(abbreviation, start_col, colorname, csv_file_path, column_number, plot_save_path, save_dpi, save, show, encoding='ISO-8859-1'):
    """
    Reads data from a CSV file, calculates averages and variances for digit values in specified columns, 
    and plots a bar chart with customized labels and colors.

    Parameters:
    - abbreviation (str): Abbreviation for plot title.
    - start_col (int): The index of the starting column for calculations.
    - colorname (str): The color name for the bars in the chart.
    - csv_file_path (str): Path to the CSV file to read.
    - column_number (int): Number of columns to consider for mean and variance calculation.
    - plot_save_path (str): Path where the plot will be saved.
    - save_dpi (int): DPI for the saved plot.
    - save (bool): Indicates whether to save the plot.
    - show (bool): Indicates whether to display the plot.
    - encoding (str): Encoding of the CSV file (default is 'ISO-8859-1').

    Returns:
    None
    """


    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path, encoding=encoding)
        
        # Get the data from the starting column
        columns = df.iloc[:, start_col]
        first_row = columns.iloc[0] if not columns.empty else "Unknown"
        
        # Initialize list to store averages, variances, and column names
        stats_list = []

        # Loop through the specified columns for mean and variance calculation
        for i in range(column_number):
            # Get the data for the current column
            column = df.iloc[:, start_col + i]
            column_data = column.iloc[2:]  # Skip the first two rows
            
            # Get the column name (from the first row in the column)
            column_name = column.iloc[1] if not column.empty else "Unknown"
            
            # Convert column data to integers
            digits = [int(digit) for digit in column_data]
            
            # Calculate the mean and variance of the digits
            average = sum(digits) / len(digits)
            variance = sum((x - average) ** 2 for x in digits) / len(digits)
            
            # Append the mean, variance, and column name to the list
            stats_list.append([average, variance, column_name])

        # Convert the list to a DataFrame for easier plotting
        stats_df = pd.DataFrame(stats_list, columns=['Average', 'Variance', 'Column'])

        # Extract the second, fourth, and sixth words from the column names and format with newlines
        def create_short_label(label):
            parts = label.split()
            if len(parts) > 5:
                # Combine the necessary parts and remove periods from the 6th part
                return '\n'.join([parts[1], parts[3], parts[5].replace('.', '')])
            else:
                # Return the original label if there are not enough parts
                return label

        # Apply the functions to create and adjust short labels
        stats_df['Short_Label'] = stats_df['Column'].apply(create_short_label)


        # Plot the bar chart for averages
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.bar(stats_df['Short_Label'], stats_df['Average'], color=colorname, alpha=0.6, label='Average')

        # Add the y-axis label for the bar chart
        ax1.set_ylabel('Value', color='black')

        # Create a secondary axis for variance
        ax2 = ax1.twinx()
        ax2.plot(stats_df['Short_Label'], stats_df['Variance'], 'ks', label='Variance')  # Black square markers
        ax2.plot(stats_df['Short_Label'], stats_df['Variance'], 'k--')  # Black dotted line

        # Set the y-axis limits for both axes to [0, 5]
        ax1.set_ylim(0, 5.2)
        ax2.set_ylim(0, 2.8)

        # Add the y-axis label for the variance
        ax2.set_ylabel('Value', color='black')

        # Create a title using the first abbreviation and first row 
        # title = abbreviation.upper() + ": " + f'{first_row}'
        # plt.title(title)

        fig.tight_layout()    
        plt.legend(loc='best')

        # Save the plot
        if save is True:
            plt.savefig(plot_save_path, dpi=save_dpi, bbox_inches='tight')

        # Show the plot
        if show is True:
            plt.show()

    except Exception as e:
        # Print the error message if an exception occurs
        print(f"Error reading CSV file or plotting: {e}")


def generate_unique_number(start_col):
    """Generate a unique number based on the start column."""
    col_map = {
        87: 1,  89: 2,  91: 3,  93: 4,  95: 5,
        97: 6,  99: 7, 101: 8, 103: 9
    }
    return col_map.get(start_col, 0)  # Return 0 if the start_col isn't in the map

# Plots for multiple datasets (comparisons) for third SAT
def plot_avg_assessment_multiple_X(abbreviation_list, start_col_list, colorname_list, csv_file_path, column_number, plot_save_path, save_dpi, save, show, encoding='ISO-8859-1'):
    """
    Plots comparison of averages and variances across multiple columns for a variable number of objects.

    This function reads a CSV file and processes selected columns to calculate the average and variance 
    for each column's values. It generates bar plots with color-coded bars for averages and line plots 
    with markers for variances. It performs t-tests between two groups of columns if applicable and annotates 
    the results on the plot. Plots include unique column names and vertical lines to separate groups.

    Parameters:
    abbreviation_list (list of str): Abbreviations for plot titles.
    start_col_list (list of int): Starting column indices to process.
    colorname_list (list of str): Colors for each plot group.
    csv_file_path (str): Path to the CSV file.
    column_number (int): Number of columns to process per group.
    encoding (str, optional): CSV file encoding, default is 'ISO-8859-1'.

    Returns:
    None
    """
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path, encoding=encoding)

        # Ensure that column_number does not exceed the available columns
        max_col_index = df.shape[1] - 1  # Get the max column index

        # Initialize a list to hold all averages, variances, and labels for plotting
        combined_avg_var_and_name_list = []
        first_row_list = []
        group_data = [] # for t-test

        # Loop through each start_col and corresponding color
        for abbreviation, start_col, colorname in zip(abbreviation_list, start_col_list, colorname_list):
            avg_var_and_name_list = []
            data_for_ttest = []

            # Get the unique number for this set of columns
            unique_number = generate_unique_number(start_col)

            # Collect the averages and labels for each question
            for i in range(int(column_number)):
                col_index = start_col + i
                if col_index > max_col_index:
                    print(f"Column index {col_index} is out of bounds.")
                    continue

                column = df.iloc[:, col_index]
                first_row = abbreviation.upper() + ": " + column.iloc[0] if not df.empty else "Unknown"
                column_data = column.iloc[2:]
                column_name = column.iloc[1] if not column.empty else "Unknown"

                # Debugging output
                # print(f"Processing column: {col_index}, Name: {column_name}")

                # Ensure the column name has enough words
                split_column_name = column_name.split()
                if len(split_column_name) < 6:
                    print(f"Column name does not have enough words: {column_name}")
                    continue

                # Separate the digits
                digits = [int(digit) for digit in column_data]
                # Calculate the mean
                average = sum(digits) / len(digits)
                # Calculate the variance
                variance = sum((x - average) ** 2 for x in digits) / len(digits)

                # Modify the column name to make it unique by appending the unique number
                unique_column_name = f"{column_name} {unique_number}"

                # Append data for t-test
                data_for_ttest.append(digits)
                
                # Append the average, variance, column name, and color to the list
                avg_var_and_name_list.append([average, variance, unique_column_name, colorname, first_row])

            # Append this question's data to the combined list
            combined_avg_var_and_name_list.append(avg_var_and_name_list)
            first_row_list.append(first_row)
            group_data.append(data_for_ttest)

        # Perform t-test between groups (assuming two groups here)
        ttest_results = []
        if len(group_data) == 2:  # Ensure there are exactly two groups for comparison
            group1, group2 = group_data
            for i in range(min(len(group1), len(group2))):  # Compare each question's data
                t_stat, p_val = ttest_ind(group1[i], group2[i], equal_var=False)  # Welch’s t-test
                ttest_results.append((t_stat, p_val))

        # Reorder the data for plotting: interleave the groups
        reordered_list = []
        for i in range(int(column_number)):  # Loop through the indices within each group
            for group in combined_avg_var_and_name_list:
                if i < len(group):
                    reordered_list.append(group[i])

        # Convert the reordered list to a DataFrame
        final_df = pd.DataFrame(reordered_list, columns=['Average', 'Variance', 'Column', 'Color', 'First_Row'])

        final_df['Short_Label'] = final_df['Column'].apply(
            lambda x: f"{x.split()[1]} {x.split()[3]} {x.split()[5].replace('.', '')} {x.split()[6]}"
        )

        # Plot the bar chart for averages
        fig, ax1 = plt.subplots(figsize=((len(abbreviation_list)+10), 7))
        ax1.bar(final_df['Short_Label'], final_df['Average'], color=final_df['Color'], alpha=0.6)

        # Create a secondary axis for variance
        ax2 = ax1.twinx()
        ax2.plot(final_df['Short_Label'], final_df['Variance'], 'ks', label='Variance')  # Black square markers
        ax2.plot(final_df['Short_Label'], final_df['Variance'], 'k--', label='Variance Line')  # Black dotted line

        # Set the y-axis limits for both axes
        ax1.set_ylim(0, 5.2)
        ax2.set_ylim(0, 2.8)

        # Add labels and titles
        ax1.set_ylabel('Average Value', color='black')
        ax2.set_ylabel('Variance', color='black')
        # plt.title('Comparison of Averages and Variances')

        # Rotate x-axis labels to make them readable
        ax1.tick_params(axis='x', rotation=90)  # Apply rotation directly to the axis
        
        # Add legends for both Average and Variance
        legend_entries = [
            Line2D([0], [0], color=colorname, lw=4, label=first_row)
            for colorname, first_row in zip(colorname_list, first_row_list)
        ]
        # Add legends to the plot, placing it outside the plot area
        ax1.legend(handles=legend_entries, loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=2) # ends up above title
        # Add legend to the upper right corner in the plot area
        ax2.legend(loc='upper right')
        # Ensure everything is laid out correctly
        plt.tight_layout()

        
        # Define positions for annotations
        positions = {
            '1': (0.195, 0.00),
            '2': (0.29, 0.00),
        }
        # Annotate Q2
        q1_results = ttest_results[:1]
        for i, (t_stat, p_val) in enumerate(q1_results):
            plt.figtext(positions['1'][0], positions['1'][1] - i * 0.03, 
                        f'1: t-stat={t_stat:.2f}, p-val={p_val:.3f}', fontsize=14, ha='left')
        # Annotate Q2
        q2_results = ttest_results[1:2]
        for i, (t_stat, p_val) in enumerate(q2_results):
            plt.figtext(positions['2'][0] + 0.33, positions['2'][1] - i * 0.03, 
                        f'2: t-stat={t_stat:.2f}, p-val={p_val:.3f}', fontsize=14, ha='left')
        
        # Draw vertical lines between x-axis labels
        num_labels = len(final_df['Short_Label'])
        for i in range(len(colorname_list), num_labels, len(colorname_list)):
            # Draw vertical line in the plot area
            ax1.axvline(x=i - 0.5, color='gray', linestyle='--', linewidth=0.8)  # Adjust ymax to extend below the x-axis labels
        
        if save is True:
            plt.savefig(plot_save_path, dpi=save_dpi, bbox_inches='tight')

        # Show the plot
        if show is True:
            plt.show()

    except Exception as e:
        print(f"Error reading CSV file or plotting: {e}")

# Gloabal settings for all figures
mpl.rcParams.update({'font.size': 16,  # for normal text
                     'axes.labelsize': 16,  # for axis labels
                     'axes.titlesize': 16})  # for title

