Github：[deepakdeepu8978-USAccidents](https://github.com/RainBowT0506/deepakdeepu8978-USAccidents)

Kaggel：[how Severity the Accidents is ?](https://www.kaggle.com/code/deepakdeepu8978/how-severity-the-accidents-is/comments)

# 認識數據
![image](https://hackmd.io/_uploads/rkgrNdqbA.png)

# 找出該欄位中所有不重複的值
## 來源
```
train_df.Source.unique()
```

```
array(['MapQuest', 'MapQuest-Bing', 'Bing'], dtype=object)
```

## 州
```
states = train_df.State.unique()
```
```
State ：
['OH' 'WV' 'CA']
```

# 按州可視化事故分佈
![image](https://hackmd.io/_uploads/S13DUuqZR.png)

# 每列中缺失值的數量
![image](https://hackmd.io/_uploads/SkTPgY5bR.png)

# 可視化起點位置關係
![image](https://hackmd.io/_uploads/ryX6gK9ZR.png)

# 視覺化終點位置關係
![image](https://hackmd.io/_uploads/SJdfWKqbA.png)

# 最容易發生事故的 5 種天氣狀況
![image](https://hackmd.io/_uploads/ry9zftc-C.png)

# 特徵工程
## 建立 dtype 資料框
![image](https://hackmd.io/_uploads/r1Z3zK9ZR.png)

## 按類型聚合資料類型計數
![image](https://hackmd.io/_uploads/SJEV7K9ZA.png)

## 辨識高度缺失列
![image](https://hackmd.io/_uploads/HydFmt5WA.png)

## 按缺失計數篩選列
```
['TMC',
 'End_Lat',
 'End_Lng',
 'Number',
 'Wind_Chill(F)',
 'Wind_Speed(mph)',
 'Precipitation(in)']
```

# 變數的相關係數
![image](https://hackmd.io/_uploads/SkVDIK9ZR.png)


## 分析唯一值
```
Turning_Loop 1
Visibility(mi) 68
Pressure(in) 311
Humidity(%) 98
Temperature(F) 769
TMC 22
```

## 選擇顯著相關性
![image](https://hackmd.io/_uploads/HJ-mYtc-0.png)


# 可視化
## 可視化相關熱圖
![image](https://hackmd.io/_uploads/ry8PKt5WA.png)
![image](https://hackmd.io/_uploads/ByFB5YqW0.png)

## 可視化特徵分佈
![image](https://hackmd.io/_uploads/SyQKsKcbA.png)
## 可視化嚴重性分佈百分比
![image](https://hackmd.io/_uploads/BJ3ue95bR.png)
## 可視化風寒的嚴重程度
![image](https://hackmd.io/_uploads/Symalc9bR.png)

## 可視化嚴重性舒適度分佈
![image](https://hackmd.io/_uploads/BJhfbq9W0.png)


## 可視化風寒強度分佈
![image](https://hackmd.io/_uploads/S1lDbccZ0.png)

## 可視化嚴重性交叉分佈
![image](https://hackmd.io/_uploads/Skgs-qqWC.png)

## 可視化嚴重程度連結分佈
![image](https://hackmd.io/_uploads/rkpAZ5qWR.png)


## 可視化嚴重程度的交通號誌分佈
![image](https://hackmd.io/_uploads/ryrGfccbC.png)

## 可視化特徵重要性
![image](https://hackmd.io/_uploads/S1lKzc5bC.png)

## 可視化 xgboost 特徵重要性
![image](https://hackmd.io/_uploads/HyJBSq5-R.png)
