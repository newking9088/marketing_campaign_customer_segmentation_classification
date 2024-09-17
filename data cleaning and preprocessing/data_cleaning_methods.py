import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy.stats import chi2_contingency
from sklearn.preprocessing import OneHotEncoder

def cramers_v(x, y):
    """Calculate Cramér's V statistic for categorical-categorical association."""
    confusion_matrix = pd.crosstab(x, y)
    chi2, p, dof, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    cramers_v_value = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
    return cramers_v_value, chi2, p, dof

def cat_feature_associations(df, target=None):
    """
    Calculate Cramér's V for each combination of categorical features and include chi-squared test results.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the features and target.
    target (str, optional): The name of the target column. If provided, calculate associations with the target.
    
    Returns:
    pd.DataFrame: A DataFrame with the column names, Cramér's V values, chi-squared statistic, p-values, and degrees of freedom.
    """
    categorical_columns = df.select_dtypes(include=['object']).columns
    results = []

    if target:
        for col in categorical_columns:
            if col != target:
                V, chi2, p, dof = cramers_v(df[col], df[target])
                results.append({'Column1': col, 'Column2': target, 'Cramér\'s V': V, 
                 'p-value': p, 'Degrees of Freedom': dof})
    else:
        for i, col1 in enumerate(categorical_columns):
            for col2 in categorical_columns[i+1:]:
                V, chi2, p, dof = cramers_v(df[col1], df[col2])
                results.append({'Column1': col1, 'Column2': col2, 
                'Cramér\'s V': V, 'p-value': p, 'Degrees of Freedom': dof})
    
    results_df = pd.DataFrame(results)

    return results_df

def cramers_v(x, y):
    """Calculate Cramér's V for two categorical variables."""
    confusion_matrix = pd.crosstab(x, y)
    chi2, p, dof, expected = chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    minDim = min(confusion_matrix.shape) - 1
    V = np.sqrt((chi2 / n) / minDim)
    return V, p

def filter_weak_associations(df, target):
    """
    Calculate Cramér's V for each categorical feature with the target and return a DataFrame 
    with the column names, Cramér's V values, and p-values.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the features and target.
    target (str): The name of the target column.
    
    Returns:
    pd.DataFrame: A DataFrame with the column names, Cramér's V values, and p-values.
    """
    categorical_columns = df.select_dtypes(include=['object']).columns
    results = []

    for col in categorical_columns:
        if col != target:
            V, p = cramers_v(df[col], df[target])
            results.append({'Column': col, 'Cramér\'s V': V, 'p-value': p})
    
    results_df = pd.DataFrame(results)
    return results_df


def merge_small_categories(df, column, threshold = 0.05, min_categories = 4, categories_under_threshold = 2):
    """
    Replace categories in a given column with 'Others' if they account for less than the specified 
    threshold and if the total number of unique categories is greater than the specified number and 
    at least two categories have a normalized value less than the threshold.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the column.
    column (str): The name of the column to process.
    threshold (float): The threshold for the minimum proportion of each category (default is 0.05).
    min_categories (int): The minimum number of unique categories required to consider replacing 
    small categories (default is 4).
    categories_under_threshold (int): The minimum number of categories that must have a normalized value 
    less than the threshold to trigger the replacement (default is 2).
    
    Returns:
    pd.DataFrame: The DataFrame with small categories replaced by 'Others'.
    """
    # Lets create a copy
    df_copy = df.copy(deep = True)

    # Calculate the proportion of each category
    category_proportions = df_copy[column].value_counts(normalize = True)
    
    # Identify small categories
    small_categories = category_proportions[category_proportions < threshold].index
    
    # Check if the total number of unique categories is greater than the specified number
    if len(category_proportions) >= min_categories:
        # Check if at least the specified number of categories have a normalized value less than the threshold
        if (category_proportions < threshold).sum() >= categories_under_threshold:
            # Replace small categories with 'Others' using map and mask
            df_copy[column] = df_copy[column].map(lambda x: 'Others' if x in small_categories else x)
    
    return df_copy

def plot_kde_before_after(df, winsorized_df, exclude_columns=None):
    """
    Plot KDE for all numeric columns before and after winsorization and print Wilcoxon rank-sum test results.
    
    Parameters:
    df (pd.DataFrame): The original DataFrame.
    winsorized_df (pd.DataFrame): The winsorized DataFrame.
    exclude_columns (list): List of columns to exclude from plotting.
    """
    numeric_columns = df.select_dtypes(include='number').columns
    plot_columns = set(numeric_columns) - set(exclude_columns) if exclude_columns else numeric_columns
    
    for column in plot_columns:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Stacked KDE plots
        sns.kdeplot(df[column], ax=ax, label='Original', fill=True, alpha=0.5)
        sns.kdeplot(winsorized_df[column], ax=ax, label='Winsorized', fill=True, alpha=0.5)
        ax.set_title(f'Stacked KDE Plot for {column}', fontweight = 'bold')
        ax.set_xlabel(f"{column.capitalize()}", fontweight = 'bold')
        ax.set_ylabel(f"Density", fontweight = 'bold')
        ax.legend(labelcolor = 'linecolor')

        plt.tight_layout()
        plt.show()

        # Wilcoxon rank-sum test
        u_stat, p_value = mannwhitneyu(df[column], winsorized_df[column])
        print(f"Wilcoxon rank-sum test for '{column}': U Statistic: {u_stat:.4f}, P-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"The distributions of '{column}' are significantly different (p < 0.05).")
        else:
            print(f"The distributions of '{column}' are not significantly different (p >= 0.05).")

# compute the vif for all given features
def compute_vif(df):
    
    X = df.select_dtypes(include = 'number')
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1
    
    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable']!='intercept']
    return vif

def replace_small_categories_with_mode(df, column, threshold = None):
    # Lets make a df copy
    df_copy = df.copy()

    # Calculate the proportions of each category
    category_proportions = df_copy[column].value_counts(normalize = True)
    
    if threshold is None:
        # Identify the category with the smallest proportion
        smallest_category = category_proportions.idxmin()
        small_categories = [smallest_category]
    else:
        # Identify small categories based on the given threshold
        small_categories = category_proportions[category_proportions <= threshold].index
    
    # Find the most frequent category
    most_frequent_category = category_proportions.idxmax()
    
    # Replace small categories with the most frequent category if number of 
    # categories in geater than 2
    if len(category_proportions) > 2:
        df_copy[column] = df_copy[column].map(lambda x: most_frequent_category if 
        x in small_categories else x)
    
    return df_copy


def get_category_cols_info(df):
    # store data into a dictionary
    value_counts_dict = {}

    # get categorical columns
    categorical_columns = df.select_dtypes(include = ['object']).columns

    # Get the value counts of each category for a given categorical    
    for col in categorical_columns:
        value_counts_dict[col] = df[col].value_counts().to_dict()

    # Convert the dictionary to a DataFrame
    multiindex_df = pd.DataFrame.from_dict(value_counts_dict, orient='index')\
    .stack().reset_index()

    # Rename the columns for clarity
    multiindex_df.columns = ['Category', 'Value', 'Count']

    # Set MultiIndex
    multiindex_df.set_index(['Category', 'Value'], inplace=True)

    return multiindex_df

def get_category_cols_info(df):
    # Store data into a dictionary
    value_counts_dict = {}
    percentage_dict = {}

    # Get categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Get the value counts and percentages of each category for a given categorical column
    for col in categorical_columns:
        counts = df[col].value_counts()
        percentages = df[col].value_counts(normalize=True) * 100
        value_counts_dict[col] = counts.to_dict()
        percentage_dict[col] = percentages.to_dict()

    # Convert the dictionaries to DataFrames
    counts_df = pd.DataFrame.from_dict(value_counts_dict, orient='index').stack().reset_index()
    percentages_df = pd.DataFrame.from_dict(percentage_dict, orient='index').stack().reset_index()

    # Rename the columns for clarity
    counts_df.columns = ['Category', 'Value', 'Count']
    percentages_df.columns = ['Category', 'Value', 'Percentage']

    # Merge the counts and percentages DataFrames
    merged_df = pd.merge(counts_df, percentages_df, on=['Category', 'Value'])

    # Set MultiIndex
    merged_df.set_index(['Category', 'Value'], inplace=True)

    return merged_df

def plot_counts_and_normalized_counts(data: pd.DataFrame, column: str, target:str = 'Subscription' , 
    figsize: tuple = (16, 8), show_counts_table: bool = True) -> None:
    """
    Plots counts and normalized counts of a specified column with a hue.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data.
    - column (str): The categorical column to plot counts and normalized counts for.
    - target (str): The categorical target column.
    - figsize (tuple): The size of the figure (default is (16, 8)).

    Returns:
    - None: Displays the plot with counts and normalized counts.
    """
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = figsize, sharey = True)

    # Group by the specified column and hue, and count occurrences
    counts = data.groupby([column, target]).size().sort_values(ascending = False).reset_index(name ='target_counts')

    # Calculate the total counts for each hue category
    total_counts = data.groupby(column).size().sort_values(ascending = False).reset_index(name = 'total_category_counts')

    # Merge the total counts with the counts
    counts = counts.merge(total_counts, on = column)

    if show_counts_table:
        print(counts)

    # Calculate the normalized counts
    counts['normalized_counts'] = counts['target_counts'] * 100.0/ counts['total_category_counts']

    # Plot using sns.barplot()
    sns.barplot(data=counts, y = column, x = 'target_counts', hue = target, ax = ax1)
    sns.barplot(data=counts, y=column, x = 'normalized_counts', hue = target, legend = False, ax = ax2)

    # Adjust layout and labels
    plt.tight_layout()
    ax1.set_ylabel(None)
    ax2.set_ylabel(None)

    # Hide all spines for both subplots
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Set title and labels
    ax1.set_xlabel("Category Counts", fontdict={'weight': 'bold'})
    ax2.set_xlabel("Category Subscription Percentage", fontdict={'weight': 'bold'})
    plt.suptitle(f"Counts and Percentage of Subscription for each Category in {column.capitalize()}")
    plt.subplots_adjust(top = 0.92)

    # Add legend to ax1
    ax1.legend(labelcolor='linecolor', title = target , loc = 'lower right')

    plt.show()

def plot_numeric_distribution(data: pd.DataFrame, column: str, target:str = 'Subscription', 
    multiple:str = 'stack', kde: bool = True, figsize: tuple = (16, 8)) -> None:
    """
    Plots histogram and normalized histogram of a specified column with a hue.

    Parameters:
    - data (pd.DataFrame): The DataFrame containing the data.
    - column (str): The numerical column to histogram and normalized histogram for.
    - target (str): The categorical target column which is used as hue.
    - multiple (str): Specifies how multiple histograms are arranged (default 'stack').Other
    options are 'dodge', 'layer' and 'fill'.
    - kde (bool): Whether to plot kde on top of histogram (default True).
    - figsize (tuple): The size of the figure (default is (16, 8)).

    Returns:
    - None: Displays the plot with hostogram and normalized histogram for numeric column.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16, 8), sharey = True)


    sns.histplot(data = data, y = column, hue = target, kde = kde, legend = False,
                alpha=0.6, multiple = multiple, ax = ax1)

    # Normalize histogram independently for each type of Subscription
    sns.histplot(data = data, y = column, hue = target, kde = kde, alpha=0.6,
                multiple=multiple, stat='percent', common_norm=False, ax=ax2)

    plt.tight_layout()

    # Hide all spines for both subplots
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_visible(False)

   # Set title and labels
    ax1.set_ylabel(column.capitalize(), fontdict={'weight': 'bold'})
    ax1.set_xlabel("Subscription Counts", fontdict={'weight': 'bold'})
    ax2.set_xlabel("Subscription Percentage", fontdict={'weight': 'bold'})
   

    plt.suptitle(f"Distribution of Counts and Percentage of Normalized Subscription for {column.capitalize()}")
    plt.subplots_adjust(top = 0.92)
    plt.show()

def one_hot_encode(df):
    """
    Perform one-hot encoding on all categorical features in the DataFrame 
    using sklearn's OneHotEncoder.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the features.
    
    Returns:
    pd.DataFrame: The DataFrame with one-hot encoded columns.
    """
    # Select categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    # Initialize OneHotEncoder with drop='first' to drop the first category
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    
    # Fit and transform the categorical columns
    one_hot_encoded = encoder.fit_transform(df[categorical_columns])
    
    # Create a DataFrame with the one-hot encoded columns
    one_hot_df = pd.DataFrame(one_hot_encoded, columns = encoder.get_feature_names_out(categorical_columns))
    
    # Concatenate the one-hot encoded DataFrame with the original DataFrame
    df_encoded = pd.concat([df.drop(categorical_columns, axis = 1), one_hot_df], axis = 1)
    
    return df_encoded





