import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set overall aesthetic style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['font.sans-serif'] = 'Segoe UI', 'DejaVu Sans', 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def generate_comparative_plots():
    print("Reading data for comparative analysis...")
    raw_df = pd.read_csv('Titanic_Dataset.csv')
    cleaned_df = pd.read_csv('Titanic_Dataset_cleaned.csv')
    
    # Enrich raw df with engineered features for intuitive readable labels
    plot_df = raw_df.copy()
    plot_df['Survival_Label'] = plot_df['Survived'].map({0: 'Died', 1: 'Survived'})
    plot_df['Pclass_Label'] = plot_df['Pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})
    plot_df['Sex_Label'] = plot_df['Sex'].str.capitalize()
    plot_df['FamilySize'] = cleaned_df['FamilySize']
    plot_df['IsAlone'] = cleaned_df['IsAlone'].map({1: 'Alone', 0: 'With Family'})
    plot_df['Has_Cabin'] = cleaned_df['Has_Cabin'].map({1: 'Cabin Allocated', 0: 'No Cabin'})
    
    # Impute Age for plot display (from cleaned_df)
    plot_df['Cleaned_Age'] = cleaned_df['Age']
    
    palette = {'Died': '#d9534f', 'Survived': '#2ca02c'}
    
    os.makedirs('plots', exist_ok=True)
    artifact_dir = r'C:\Users\RIO\.gemini\antigravity-ide\brain\94d57e6d-5350-4125-be1c-11902e84351a'

    # -------------------------------------------------------------
    # 1. Comprehensive Multi-Panel Dashboard (2x3 Grid)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Titanic Survival Analysis: Parameter Comparison & Insights', fontsize=18, fontweight='bold', y=0.98)

    # Subplot 1: Survival by Class & Gender
    sns.barplot(
        data=plot_df, x='Pclass_Label', y='Survived', hue='Sex_Label',
        palette={'Male': '#4a90e2', 'Female': '#e83e8c'},
        errorbar=None, ax=axes[0, 0]
    )
    axes[0, 0].set_title('1. Survival Rate by Class & Gender', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Passenger Class')
    axes[0, 0].set_ylabel('Survival Rate')
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].legend(title='Gender', loc='upper right')
    for p in axes[0, 0].patches:
        height = p.get_height()
        if height > 0:
            axes[0, 0].annotate(f'{height:.1%}', (p.get_x() + p.get_width() / 2., height),
                                ha='center', va='bottom', fontsize=9, xytext=(0, 2),
                                textcoords='offset points')

    # Subplot 2: Age Distribution by Survival Status
    sns.kdeplot(
        data=plot_df, x='Cleaned_Age', hue='Survival_Label', common_norm=False,
        fill=True, alpha=0.4, palette=palette, ax=axes[0, 1]
    )
    axes[0, 1].set_title('2. Age Density: Survivors vs Non-Survivors', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Age (Years)')
    axes[0, 1].set_ylabel('Density')

    # Subplot 3: Fare vs Survival by Class (capped for visual clarity)
    sns.boxplot(
        data=plot_df, x='Pclass_Label', y='Fare', hue='Survival_Label',
        palette=palette, showfliers=False, ax=axes[0, 2]
    )
    axes[0, 2].set_title('3. Fare Distribution across Classes & Survival', fontsize=13, fontweight='bold')
    axes[0, 2].set_xlabel('Passenger Class')
    axes[0, 2].set_ylabel('Fare (GBP)')
    axes[0, 2].legend(title='Status', loc='upper right')

    # Subplot 4: Family Size vs Survival Rate
    fam_stat = plot_df.groupby('FamilySize')['Survived'].agg(['mean', 'count']).reset_index()
    sns.barplot(data=fam_stat, x='FamilySize', y='mean', color='#5c6bc0', ax=axes[1, 0])
    axes[1, 0].set_title('4. Survival Rate by Family Size', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Family Size (SibSp + Parch + 1)')
    axes[1, 0].set_ylabel('Survival Rate')
    axes[1, 0].set_ylim(0, 1.0)
    for p in axes[1, 0].patches:
        h = p.get_height()
        if h > 0:
            axes[1, 0].annotate(f'{h:.1%}', (p.get_x() + p.get_width() / 2., h),
                                ha='center', va='bottom', fontsize=9, xytext=(0, 2),
                                textcoords='offset points')

    # Subplot 5: Embarkation Port & Cabin Assignment Impact
    sns.barplot(
        data=plot_df, x='Embarked', y='Survived', hue='Has_Cabin',
        palette={'No Cabin': '#e74c3c', 'Cabin Allocated': '#3498db'},
        errorbar=None, ax=axes[1, 1]
    )
    axes[1, 1].set_title('5. Survival by Port & Cabin Allocation', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Port of Embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)')
    axes[1, 1].set_ylabel('Survival Rate')
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].legend(title='Cabin Status', loc='upper left')

    # Subplot 6: Correlation Heatmap of Numeric & Engineered Features
    corr_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Has_Cabin', 'FamilySize', 'IsAlone']
    corr_matrix = cleaned_df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-0.6, vmax=0.6,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=axes[1, 2])
    axes[1, 2].set_title('6. Correlation Matrix of Key Features', fontsize=13, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    dashboard_path = os.path.join('plots', 'titanic_comparative_dashboard.png')
    plt.savefig(dashboard_path, dpi=200)
    plt.close()
    print(f"Saved main dashboard to: {dashboard_path}")
    
    # -------------------------------------------------------------
    # 2. Individual Focused Plots
    # -------------------------------------------------------------
    # Individual Plot A: Gender and Pclass
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=plot_df, x='Pclass_Label', y='Survived', hue='Sex_Label',
        palette={'Male': '#4a90e2', 'Female': '#e83e8c'}, errorbar=None
    )
    plt.title('Survival Rate: Intersection of Class & Gender', fontsize=14, fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Survival Rate')
    plt.ylim(0, 1.05)
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f'{h:.1%}', (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 3),
                        textcoords='offset points')
    plt.tight_layout()
    plot_a = os.path.join('plots', 'class_gender_survival.png')
    plt.savefig(plot_a, dpi=150)
    plt.close()

    # Individual Plot B: Age vs Survival with Titles
    plt.figure(figsize=(9, 5))
    sns.violinplot(
        data=plot_df, x='Pclass_Label', y='Cleaned_Age', hue='Survival_Label',
        split=True, palette=palette, inner='quart'
    )
    plt.title('Age Distribution by Class and Survival (Violin Plot)', fontsize=14, fontweight='bold')
    plt.xlabel('Passenger Class')
    plt.ylabel('Age (Years)')
    plt.tight_layout()
    plot_b = os.path.join('plots', 'age_class_survival_violin.png')
    plt.savefig(plot_b, dpi=150)
    plt.close()

    # Individual Plot C: Correlation Heatmap standalone
    plt.figure(figsize=(9, 7))
    sns.heatmap(cleaned_df.corr(), annot=False, cmap='coolwarm', vmin=-0.6, vmax=0.6, linewidths=0.5)
    plt.title('Full Dataset Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_c = os.path.join('plots', 'full_correlation_heatmap.png')
    plt.savefig(plot_c, dpi=150)
    plt.close()

    # Copy all plots to the artifact directory for embedding in reports
    for fname in ['titanic_comparative_dashboard.png', 'class_gender_survival.png', 'age_class_survival_violin.png', 'full_correlation_heatmap.png']:
        src = os.path.join('plots', fname)
        dst = os.path.join(artifact_dir, fname)
        shutil.copy(src, dst)
        print(f"Copied {fname} to artifact directory.")

    print("\nAll comparative plots successfully generated and saved!")

if __name__ == '__main__':
    generate_comparative_plots()
