import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Plot settings
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output

# Load data
data = pd.read_csv("C:\\Users\\Mahendra\\Downloads\\Crime_Incidents_in_2024.csv")
data.columns = data.columns.str.strip()

# -------------------- CLEANING --------------------
# Drop fully null columns
data = data.drop(columns=['OCTO_RECORD_ID'])

# Fill numeric missing values with median
data['X'] = data['X'].fillna(data['X'].median())
data['Y'] = data['Y'].fillna(data['Y'].median())

if data['SHIFT'].isnull().sum() > 0:
    data['SHIFT'] = data['SHIFT'].fillna(data['SHIFT'].mode().iloc[0])

# For OFFENSE
if data['OFFENSE'].isnull().sum() > 0:
    data['OFFENSE'] = data['OFFENSE'].fillna(data['OFFENSE'].mode().iloc[0])

# For METHOD
if 'METHOD' in data.columns and data['METHOD'].isnull().sum() > 0:
    data['METHOD'] = data['METHOD'].fillna(data['METHOD'].mode().iloc[0])

# For BLOCK (if it's a string/categorical column)
if 'BLOCK' in data.columns and data['BLOCK'].isnull().sum() > 0:
    data['BLOCK'] = data['BLOCK'].fillna(data['BLOCK'].mode().iloc[0])

# -------------------- CREATE HOUR COLUMN --------------------
# Convert 'REPORT_DAT' to datetime first
data['REPORT_DAT'] = pd.to_datetime(data['REPORT_DAT'], errors='coerce')

# Now you can safely extract the hour
data['HOUR'] = data['REPORT_DAT'].dt.hour

# Fill missing 'HOUR' values with the median of the 'HOUR' column
data['HOUR'] = data['HOUR'].fillna(data['HOUR'].median())

# Convert date columns
data['REPORT_DAT'] = pd.to_datetime(data['REPORT_DAT'], errors='coerce')
data['START_DATE'] = pd.to_datetime(data['START_DATE'], errors='coerce')
data['END_DATE'] = pd.to_datetime(data['END_DATE'], errors='coerce')

# Remove timezone to avoid PeriodArray warning
data['REPORT_DAT'] = data['REPORT_DAT'].dt.tz_localize(None)

# BASIC INFO
print("\n=== BASIC INFO ===")
print(data.info())
print("\nSummary Stats:")
print(data.describe(include='all'))

# INCIDENTS OVER TIME
plt.figure()
data['REPORT_DAT'].dt.date.value_counts().sort_index().plot()
plt.title("Daily Crime Incidents in 2024")
plt.xlabel("Date")
plt.ylabel("Incident Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# CRIME TYPE DISTRIBUTION
plt.figure()
sns.countplot(y='OFFENSE', hue='OFFENSE', data=data,
              order=data['OFFENSE'].value_counts().head(10).index,
              palette="viridis", legend=False)
plt.title("Top 10 Crime Types")
plt.xlabel("Number of Incidents")
plt.ylabel("Offense Type")
plt.tight_layout()
plt.show()

# SHIFT ANALYSIS
plt.figure()
sns.countplot(x='SHIFT', hue='SHIFT', data=data, palette='Set2', legend=False)
plt.title("Incidents by Shift (Time of Day)")
plt.xlabel("Shift")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# METHOD ANALYSIS
plt.figure()
sns.countplot(x='METHOD', hue='METHOD', data=data,
              order=data['METHOD'].value_counts().index,
              palette='muted', legend=False)
plt.title("Incidents by Method")
plt.xlabel("Method Used")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# OFFENSE BY DISTRICT
plt.figure()
sns.countplot(data=data, x='DISTRICT', hue='OFFENSE',
              order=sorted(data['DISTRICT'].dropna().unique()))
plt.title("Offense Types Across Police Districts")
plt.xlabel("District")
plt.ylabel("Incident Count")
plt.legend(title='Offense', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# HEATMAP: OFFENSE VS SHIFT
offense_shift = pd.crosstab(data['OFFENSE'], data['SHIFT'])
plt.figure(figsize=(12, 6))
sns.heatmap(offense_shift, annot=True, fmt='d', cmap='YlGnBu')
plt.title("Heatmap of Offense Type vs Shift")
plt.tight_layout()
plt.show()

# BOX PLOT: INCIDENTS BY SHIFT
plt.figure()
sns.boxplot(x='SHIFT', y=data['XBLOCK'], data=data)
plt.title("Boxplot of XBLOCK by Shift")
plt.xlabel("Shift")
plt.ylabel("XBLOCK")
plt.tight_layout()
plt.show()

# PAIRPLOT: First 30 records for numerical relationships
df = data.head(30)
sns.pairplot(data=df, hue="SHIFT", palette="pastel")
plt.suptitle("Pairwise Plot of Coordinates by Shift", y=1.02)
plt.show()

# HISTOGRAM: Latitude and Longitude
for col in ['XBLOCK', 'YBLOCK']:
    plt.figure()
    sns.histplot(data[col].dropna(), bins=30, kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# CATPLOT: Crimes by Ward
sns.catplot(x='WARD', kind='count', data=data, height=5, aspect=2)
plt.title("Crime Count by Ward")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# LINEPLOT: Monthly Crime Trend
data['Month'] = data['REPORT_DAT'].dt.to_period('M').astype(str)
monthly_trend = data['Month'].value_counts().sort_index()
plt.figure()
sns.lineplot(x=monthly_trend.index, y=monthly_trend.values, marker='o')
plt.title("Monthly Crime Incidents in 2024")
plt.xlabel("Month")
plt.ylabel("Incident Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# SCATTERPLOT: Location of Incidents
plt.figure()
sns.scatterplot(x='XBLOCK', y='YBLOCK', data=data, hue='SHIFT', palette='Set2', alpha=0.6)
plt.title("Crime Locations Colored by Shift")
plt.xlabel("XBLOCK")
plt.ylabel("YBLOCK")
plt.legend()
plt.tight_layout()
plt.show()

# PIE CHART: Proportion of Crime Methods
plt.figure()
method_counts = data['METHOD'].value_counts()
plt.pie(method_counts, labels=method_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Distribution of Crime Methods")
plt.tight_layout()
plt.show()

# Z-TEST: GUN vs KNIFE incidents (by count)
def perform_z_test(group1, group2, label1, label2):
    x1 = group1.shape[0]
    x2 = group2.shape[0]
    n = x1 + x2
    p1 = x1 / n
    p2 = x2 / n
    p = (x1 + x2) / (2 * n)
    se = np.sqrt(p * (1 - p) * (1/n + 1/n))
    z = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    print(f"\n===== Z-TEST: {label1} vs {label2} =====")
    print(f"Count {label1}: {x1}")
    print(f"Count {label2}: {x2}")
    print(f"Z-Score: {z:.4f}")
    print(f"P-Value: {p_value:.4f}")
    if p_value < 0.05:
        print("Conclusion: Significant difference (p < 0.05)")
    else:
        print("Conclusion: No significant difference (p ≥ 0.05)")

# Apply Z-test
gun_incidents = data[data['METHOD'].str.upper() == 'GUN']
knife_incidents = data[data['METHOD'].str.upper() == 'KNIFE']
perform_z_test(gun_incidents, knife_incidents, 'GUN', 'KNIFE')
