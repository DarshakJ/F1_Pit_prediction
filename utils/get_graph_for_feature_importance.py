import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

feature_importance_path = "Fature_importance"

def analyze_feature_variation(
    df,
    target='PitNextLap',
    group_0=0,
    group_1=1,
    columns=None
):
    """
    Compare feature distributions between target groups.

    Returns features ranked by absolute Cohen's d.
    """

    if columns is None:
        columns = [
            'Year',
            'LapNumber',
            'Stint',
            'TyreLife',
            'Position',
            'LapTime (s)',
            'LapTime_Delta',
            'Cumulative_Degradation',
            'RaceProgress',
            'Position_Change',
            'MaxTireLife',
            'TireRemainingLife',
            'TireAgePct',
            'TireRemainingPct',
            'RaceLaps',
            'LapsRemaining',
            'Prev_Position',
            'LapTimeAvg_3'
        ]

    results = []

    for col in columns:

        if col not in df.columns:
            continue

        d0 = df.loc[df[target] == group_0, col].dropna()
        d1 = df.loc[df[target] == group_1, col].dropna()

        if len(d0) < 2 or len(d1) < 2:
            continue

        mean_0 = d0.mean()
        mean_1 = d1.mean()

        median_0 = d0.median()
        median_1 = d1.median()

        std_0 = d0.std()
        std_1 = d1.std()

        # Pooled standard deviation
        pooled_std = np.sqrt(
            (
                (len(d0) - 1) * std_0**2 +
                (len(d1) - 1) * std_1**2
            )
            /
            (len(d0) + len(d1) - 2)
        )

        # Cohen's d
        cohens_d = (
            (mean_1 - mean_0) / pooled_std
            if pooled_std != 0
            else 0
        )

        results.append({
            'Feature': col,
            'Mean_0': mean_0,
            'Mean_1': mean_1,
            'Mean_Diff': mean_1 - mean_0,
            'Median_0': median_0,
            'Median_1': median_1,
            'Median_Diff': median_1 - median_0,
            'Std_0': std_0,
            'Std_1': std_1,
            'Cohens_d': cohens_d,
            'Abs_Cohens_d': abs(cohens_d)
        })

    result = pd.DataFrame(results)

    if not result.empty:
        result = (
            result
            .sort_values('Abs_Cohens_d', ascending=False)
            .reset_index(drop=True)
        )

    return result


def variation_level(d):
    d = abs(d)

    if d < 0.2:
        return 'Very Small'
    elif d < 0.5:
        return 'Small'
    elif d < 0.8:
        return 'Medium'
    elif d < 1.2:
        return 'Large'
    else:
        return 'Very Large'

    

def plot_variations(variation_df,sample_type,feature_eng_name):
    variation_df['Variation'] = (
    variation_df['Cohens_d']
    .apply(variation_level)
    )

    print(
        variation_df[
            ['Feature', 'Cohens_d', 'Abs_Cohens_d', 'Variation']
        ].to_string(index=False)
    )
    plot_df = variation_df.sort_values(
    'Abs_Cohens_d',
    ascending=True
)

    plt.figure(figsize=(12, 8))

    plt.barh(
        plot_df['Feature'],
        plot_df['Abs_Cohens_d']
    )

    plt.xlabel("Absolute Cohen's d")
    plt.ylabel("Feature")
    plt.title("Feature Variation: PitNextLap = 0 vs 1")

    plt.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    plt.savefig(f"{feature_importance_path}/{sample_type}/{feature_eng_name}_feature_variations.png", dpi=300, bbox_inches="tight")
    plt.show()

def get_graph(df,columns,sample_type,feature_eng_name):
    # Don't plot the target itself
    columns_to_plot = [c for c in columns if c != 'PitNextLap']

    df_plot = df.copy()

    for col in columns_to_plot:

        fig, axes = plt.subplots(
            1, 2,
            figsize=(14, 5),
            sharey=False
        )

        data_0 = df_plot.loc[df_plot['PitNextLap'] == 0, col].dropna()
        data_1 = df_plot.loc[df_plot['PitNextLap'] == 1, col].dropna()

        # -------------------------
        # Numeric columns
        # -------------------------
        if pd.api.types.is_numeric_dtype(df_plot[col]):

            axes[0].hist(
                data_0,
                bins=30,
                alpha=0.75
            )
            axes[0].set_title(f'{col} | PitNextLap = 0')
            axes[0].set_xlabel(col)
            axes[0].set_ylabel('Count')
            axes[0].grid(alpha=0.2)

            axes[1].hist(
                data_1,
                bins=30,
                alpha=0.75
            )
            axes[1].set_title(f'{col} | PitNextLap = 1')
            axes[1].set_xlabel(col)
            axes[1].set_ylabel('Count')
            axes[1].grid(alpha=0.2)

        # -------------------------
        # Categorical columns
        # -------------------------
        else:

            counts_0 = data_0.value_counts().head(20)
            counts_1 = data_1.value_counts().head(20)

            axes[0].bar(
                counts_0.index.astype(str),
                counts_0.values
            )
            axes[0].set_title(f'{col} | PitNextLap = 0')
            axes[0].set_xlabel(col)
            axes[0].set_ylabel('Count')
            axes[0].tick_params(axis='x', rotation=75)
            axes[0].grid(axis='y', alpha=0.2)

            axes[1].bar(
                counts_1.index.astype(str),
                counts_1.values
            )
            axes[1].set_title(f'{col} | PitNextLap = 1')
            axes[1].set_xlabel(col)
            axes[1].set_ylabel('Count')
            axes[1].tick_params(axis='x', rotation=75)
            axes[1].grid(axis='y', alpha=0.2)

        plt.suptitle(
            f'Distribution of {col} by PitNextLap',
            fontsize=14,
            fontweight='bold'
        )
        plt.savefig(f"{feature_importance_path}/{sample_type}/{feature_eng_name}_distribution_by_pitnext_{col}.png", dpi=300, bbox_inches="tight")

        plt.tight_layout()
        plt.show()


import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

def plot_category_graph(variation_df,sample_type,feature_eng_name):
    plt.figure(figsize=(10, 5))
    plot_df = variation_df.sort_values(
    'Cramers_V'
    )
    plt.barh(
        plot_df['Feature'],
        plot_df['Cramers_V']
    )

    plt.xlabel("Cramer's V")
    plt.ylabel("Feature")
    plt.title("Categorical Feature Association with PitNextLap")
    plt.savefig(f"{feature_importance_path}/{sample_type}/{feature_eng_name}_cetgorical_features_variation.png", dpi=300, bbox_inches="tight")
    plt.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    plt.show()

def categorical_variation(
    df,
    sample_type,feature_eng_name,
    target='PitNextLap',
    columns=None
):
    """
    Calculate Cramer's V for categorical features
    against a binary target.

    Higher Cramer's V = stronger association.
    """

    if columns is None:
        columns = [
            'Driver',
            'Compound',
            'Race'
        ]

    results = []

    for col in columns:

        if col not in df.columns:
            continue

        temp = df[[col, target]].dropna()

        # Crosstab
        table = pd.crosstab(
            temp[col],
            temp[target]
        )

        if table.shape[0] < 2 or table.shape[1] < 2:
            continue

        # Chi-square
        chi2, p_value, _, _ = chi2_contingency(table)

        n = table.sum().sum()
        r, k = table.shape

        # Cramer's V
        phi2 = chi2 / n

        cramer_v = np.sqrt(
            phi2 / min(k - 1, r - 1)
        )

        results.append({
            'Feature': col,
            'Cramers_V': cramer_v,
            'P_Value': p_value,
            'Categories': table.shape[0]
        })

    result = (
        pd.DataFrame(results)
        .sort_values(
            'Cramers_V',
            ascending=False
        )
        .reset_index(drop=True)
    )


    plot_category_graph(result,sample_type,feature_eng_name)
    
    return result



