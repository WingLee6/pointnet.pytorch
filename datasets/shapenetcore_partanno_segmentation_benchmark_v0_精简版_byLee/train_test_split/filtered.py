import json

# 读取 JSON 文件
with open('./shuffled_train_file_list.json', 'r') as file:
    data = json.load(file)

# 关键字列表
keywords = ['02691156', '02773838', '02954340', '02958343']  # 根据实际需要更新关键字列表

# 筛选出包含任何关键字的元素
filtered_data = [item for item in data if any(keyword in item for keyword in keywords)]

# 打印或保存筛选后的数据
print(filtered_data)

# 如果你需要将结果保存到一个新文件中，可以使用下面的代码
with open('shuffled_train_file_list_filtered.json', 'w') as file:
    json.dump(filtered_data, file, indent=4)
