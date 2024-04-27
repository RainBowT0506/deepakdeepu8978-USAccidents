import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import ensemble
import xgboost as xgb


pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

train_df = pd.read_csv('small_data.csv')

print("Train DataFrame ：")
print(train_df.shape)
print(train_df.head())

print("Source ：")
print(train_df.Source.unique())

states = train_df.State.unique()
print("State ：")
print(states)

# 按州可視化事故分佈
def visualize_accident_distribution_by_state():
    count_by_state = []
    for state in states:
        count_by_state.append(train_df[train_df['State'] == state].count()['ID'])

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.barplot(x=states, y=count_by_state, ax=ax)
    plt.show()

# 每列中缺失值的數量
def visualize_missing_data_distribution():
    missing_df = train_df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df = missing_df[missing_df['missing_count'] > 0]  # 過濾出有缺失值的欄位
    missing_df = missing_df.sort_values(by='missing_count')

    ind = np.arange(missing_df.shape[0])
    width = 0.5
    fig, ax = plt.subplots(figsize=(12, 18))
    rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
    ax.set_yticks(ind)
    ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
    ax.set_xlabel("Count of missing values")
    ax.set_title("Number of missing values in each column")
    plt.show()

# 可視化起點位置關係
def visualize_start_location_relationship():
    sns.jointplot(x=train_df.Start_Lat.values, y=train_df.Start_Lng.values, height=10)
    plt.ylabel('Start_Lat', fontsize=12)
    plt.xlabel('Start_Lng', fontsize=12)
    plt.show()

# 視覺化終點位置關係
def visualize_end_location_relationship():
    sns.jointplot(x=train_df.End_Lat.values, y=train_df.End_Lng.values, height=10)
    plt.ylabel('End_Lat', fontsize=12)
    plt.xlabel('End_Lng', fontsize=12)
    plt.show()

# 最容易發生事故的 5 種天氣狀況
def visualize_top_weather_conditions():
    fig, ax = plt.subplots(figsize=(16, 7))
    train_df['Weather_Condition'].value_counts().sort_values(ascending=False).head(5).plot.bar(width=0.5, edgecolor='k',
                                                                                               align='center',
                                                                                               linewidth=2)
    plt.xlabel('Weather_Condition', fontsize=20)
    plt.ylabel('Number of Accidents', fontsize=20)
    ax.tick_params(labelsize=20)
    plt.title('Top 5 Weather Condition for accidents', fontsize=25)
    plt.grid()
    plt.ioff()


# visualize_accident_distribution_by_state()

# visualize_missing_data_distribution()

# visualize_start_location_relationship()

# visualize_end_location_relationship()

# visualize_top_weather_conditions()

dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
print(dtype_df)

dtype_df.groupby("Column Type").aggregate('count').reset_index()
print(dtype_df)

missing_df = train_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['columns_name', 'missing_count']
missing_df['missing_ratio'] = missing_df['missing_count'] / train_df.shape[0]
missing_ratio = missing_df.loc[missing_df['missing_ratio'] > 0.777]
print(missing_ratio)

missin = missing_df.loc[missing_df['missing_count'] > 250000]
removelist = missin['columns_name'].tolist()
print(removelist)

# 提取時間特徵
# extract_time_features
train_df['Start_Time'] = pd.to_datetime(train_df['Start_Time'], errors='coerce')
train_df['End_Time'] = pd.to_datetime(train_df['End_Time'], errors='coerce')
# Extract year, month, day, hour and weekday
train_df['Year'] = train_df['Start_Time'].dt.year
train_df['Month'] = train_df['Start_Time'].dt.strftime('%b')
train_df['Day'] = train_df['Start_Time'].dt.day
train_df['Hour'] = train_df['Start_Time'].dt.hour
train_df['Weekday'] = train_df['Start_Time'].dt.strftime('%a')
# Extract the amount of time in the unit of minutes for each accident, round to the nearest integer
td = 'Time_Duration(min)'
train_df[td] = round((train_df['End_Time'] - train_df['Start_Time']) / np.timedelta64(1, 'm'))
print(train_df)

# 刪除負持續時間異常值
# remove negative time duration outliers
neg_outliers = train_df[td] <= 0
# Set outliers to NAN
train_df[neg_outliers] = np.nan
# Drop rows with negative td
train_df.dropna(subset=[td], axis=0, inplace=True)

feature_lst=['Source','TMC','Severity','Start_Lng','Start_Lat','Distance(mi)','Side','City','County','State','Timezone','Temperature(F)','Humidity(%)','Pressure(in)', 'Visibility(mi)', 'Wind_Direction','Weather_Condition','Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop','Sunrise_Sunset','Hour','Weekday', 'Time_Duration(min)']

feature_lst = [col for col in feature_lst if col in train_df.columns]

df = train_df[feature_lst].copy()
print(df.info())

def visualize_variable_correlation():
    corr_df, labels = get_correlation_dataframe()

    ind = np.arange(len(labels))
    width = 0.9
    fig, ax = plt.subplots(figsize=(12, 40))
    rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')
    ax.set_yticks(ind)
    ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
    ax.set_xlabel("Correlation coefficient")
    ax.set_title("Correlation coefficient of the variables")
    plt.show()


def get_correlation_dataframe():
    x_cols = [col for col in df.columns if col not in ['Severity'] if df[col].dtype == 'float64']
    labels = []
    values = []
    for col in x_cols:
        labels.append(col)
        values.append(np.corrcoef(df[col].values, df.Severity.values)[0, 1])
    corr_df = pd.DataFrame({'col_labels': labels, 'corr_values': values})
    corr_df = corr_df.sort_values(by='corr_values')
    return corr_df, labels


# visualize_variable_correlation()

# 分析唯一值
corr_zero_columns = ['Turning_Loop','Visibility(mi)','Pressure(in)','Humidity(%)','Temperature(F)']
for col in corr_zero_columns:
    print(col,len(df[col].unique()))

# 選擇顯著相關性
corr_df, labels = get_correlation_dataframe()
corr_df_sel = corr_df.loc[(corr_df['corr_values']>0.05) | (corr_df['corr_values'] < -0.05)]
print(corr_df_sel)


# 可視化相關熱圖
def visualize_correlation_heatmap():
    corr_df_ = corr_df_sel.col_labels.tolist()
    tem_df = df[corr_df_]
    corrmat = tem_df.corr(method='spearman')
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(corrmat, vmax=1, square=True)
    plt.title('corr map', fontsize=15)
    fig = plt.gcf()
    fig.set_size_inches(20, 20)
    fig = sns.heatmap(df.corr(), annot=True, linewidths=1, linecolor='k', square=True, mask=False, vmin=-1, vmax=1,
                      cbar_kws={"orientation": "vertical"}, cbar=True)
    plt.show()


# visualize_correlation_heatmap()

# 可視化特徵分佈
def visualize_feature_distribution():
    fig = plt.figure(figsize=(10, 10))
    fig_dims = (3, 2)
    plt.subplot2grid(fig_dims, (0, 0))
    df['Amenity'].value_counts().plot(kind='bar',
                                      title='Amenity')
    plt.subplot2grid(fig_dims, (0, 1))
    df['Crossing'].value_counts().plot(kind='bar',
                                       title='Crossing')
    plt.subplot2grid(fig_dims, (1, 0))
    df['Junction'].value_counts().plot(kind='bar',
                                       title='Junction')
    plt.subplot2grid(fig_dims, (1, 1))
    df['Junction'].value_counts().plot(kind='bar',
                                       title='Junction')


# visualize_feature_distribution()

# 可視化嚴重性分佈百分比
def visualize_severity_distribution_percentage():
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    explode = [0.1] * df['Severity'].nunique()  # 使用相同的值，這裡示範使用 0.1
    df['Severity'].value_counts().plot.pie(explode=explode, autopct='%1.1f%%', ax=ax[0], shadow=True)
    ax[0].set_title('Percentage Severity Distribution')
    ax[0].set_ylabel('Count')
    sns.countplot(x='Severity', data=df, ax=ax[1], order=df['Severity'].value_counts().index)
    ax[1].set_title('Count of Severity')
    plt.show()

# visualize_severity_distribution_percentage()

# 可視化風寒的嚴重程度
def visualize_wind_chill_severity():
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="Severity", y="Wind_Chill(F)", data=train_df)
    plt.ylabel('Wind_Chill(F)', fontsize=12)
    plt.xlabel('Severity', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.show()

# visualize_wind_chill_severity()

# 可視化嚴重性舒適度分佈
def visualize_severity_amenity_distribution():
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Severity', y='Amenity', data=train_df)
    plt.xlabel('Severity', fontsize=12)
    plt.ylabel('Amenity', fontsize=12)
    plt.show()

# visualize_severity_amenity_distribution()

# 可視化風寒強度分佈
def visualize_severity_windchill_distribution():
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Severity', y='Wind_Chill(F)', data=train_df)
    plt.xlabel('Severity', fontsize=12)
    plt.ylabel('Wind_Chill(F)', fontsize=12)
    plt.show()

# visualize_severity_windchill_distribution()


# 可視化嚴重性交叉分佈
def visualize_severity_crossing_distribution():
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Severity', y='Crossing', data=train_df)
    plt.xlabel('Severity', fontsize=12)
    plt.ylabel('Crossing', fontsize=12)
    plt.show()

# visualize_severity_crossing_distribution()

# 可視化嚴重程度連結分佈
def visualize_severity_junction_distribution():
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Severity', y='Junction', data=train_df)
    plt.xlabel('Severity', fontsize=12)
    plt.ylabel('Junction', fontsize=12)
    plt.show()

# visualize_severity_junction_distribution()

# 可視化嚴重程度的交通號誌分佈
def visualize_severity_traffic_signal_distribution():
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='Severity', y='Traffic_Signal', data=train_df)
    plt.xlabel('Severity', fontsize=12)
    plt.ylabel('Traffic_Signal', fontsize=12)
    plt.show()

# visualize_severity_traffic_signal_distribution()


df.dropna(subset=df.columns[df.isnull().mean()!=0], how='any', axis=0, inplace=True)
print(df.shape)

# 可視化特徵重要性
def visualize_feature_importance():
    train_y = df['Severity'].values
    x_cols = [col for col in df.columns if col not in ['Severity'] if df[col].dtype == 'float64']
    train_col = df[x_cols]
    fearture_name = train_col.columns.values
    model = ensemble.ExtraTreesRegressor(n_estimators=25, max_depth=30, max_features=0.3, n_jobs=-1, random_state=0)
    model.fit(train_col, train_y)
    # plot imp
    importance = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importance)[::-1][:20]
    plt.figure(figsize=(12, 12))
    plt.title("Feature importances")
    plt.bar(range(len(indices)), importance[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(indices)), fearture_name[indices], rotation='vertical')
    plt.xlim([-1, len(indices)])
    plt.show()


# visualize_feature_importance()


# 可視化 xgboost 特徵重要性
def visualize_xgboost_feature_importance():
    xgb_prames = {
        'eta': 0.05,
        'max_depth': 8,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'silent': 1,
        'seed': 0
    }
    train_y = df['Severity'].values
    x_cols = [col for col in df.columns if col not in ['Severity'] if df[col].dtype == 'float64']
    train_col = df[x_cols]
    dtrain = xgb.DMatrix(train_col, train_y, feature_names=train_col.columns.tolist())
    model = xgb.train(dict(xgb_prames, silent=0), dtrain, num_boost_round=50)
    fig, ax = plt.subplots(figsize=(12, 18))
    xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
    plt.show()


visualize_xgboost_feature_importance()