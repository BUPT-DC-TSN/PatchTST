V-5验证组：前5个月作为训练集，后5个月作为测试集



选俩月作为训练集，预测俩月后的数据，俩月后的真实数据作为测试集进行验证



uA怎么算的？



Table 2：

横着：

- V-1: 前两个月的数据预测下两个月的数据
- V-2: 前四个月的数据预测下两个月的数据
- V-3: 前六个月的数据预测下两个月的数据
- V-4: 前八个月的数据预测下两个月的数据
- **V-5: 前五个月的数据预测下五个月的数据**

竖着：

Set-com: Allan方差的倒数加权A部分中的所有值





convert.py: 根据drift.txt和CircularT.txt得到merged_data.csv

convert_set_com.py: 根据merged_data.csv得到set_com.csv

