import numpy as np
import pyarrow.parquet as pq
import os

# ==================== 配置 ====================
NUM_SENSORS = 50  # 取前 50 个传感器
INPUT_STEPS = 12  # x_t-11 ~ x_t+0
OUTPUT_STEPS = 12  # y_t+1 ~ y_t+12
INPUT_FEATS = 2   # d0=流量, d1=时间编码
OUTPUT_FEATS = 2

# 路径
base_path = os.environ.get('TRAFFIC_DATA_DIR') or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '数据集')
train_parquet = os.path.join(base_path, "train.parquet")
val_parquet   = os.path.join(base_path, "val.parquet")
test_parquet  = os.path.join(base_path, "test.parquet")
adj_npy       = os.path.join(base_path, "adj_mx.npy")

# 输出（按传感器重塑后的多维数组）
# 形状: (num_sensors, num_samples_per_sensor, time_steps, features)
train_out_x = os.path.join(base_path, "train_x.npy")  # (S, N, 12, 2)
train_out_y = os.path.join(base_path, "train_y.npy")  # (S, N, 12, 2)
val_out_x   = os.path.join(base_path, "val_x.npy")
val_out_y   = os.path.join(base_path, "val_y.npy")
test_out_x  = os.path.join(base_path, "test_x.npy")
test_out_y  = os.path.join(base_path, "test_y.npy")
adj_out     = os.path.join(base_path, "adj_50.npy")


def parquet_to_xy(in_path, out_x, out_y, num_sensors, input_steps=12, output_steps=12):
    """
    将 PEMS-BAY parquet（长格式：每行是一个 (sensor, time) 样本）
    重塑为按传感器的多维数组。
    """
    print(f"    读取: {in_path}")
    table = pq.read_table(in_path, columns=['node_id'] +
                          [f'x_t-{input_steps-1-i}_d{j}' for i in range(input_steps) for j in range(2)] +
                          [f'y_t+{i+1}_d{j}' for i in range(output_steps) for j in range(2)])
    df = table.to_pandas()
    n_rows = len(df)
    print(f"    总行数: {n_rows}")

    # 节点数
    unique_nodes = sorted(df['node_id'].unique())
    print(f"    节点数: {len(unique_nodes)}")
    selected_nodes = unique_nodes[:num_sensors]
    print(f"    取前 {num_sensors} 个节点: {selected_nodes[:5]}...")

    # 过滤
    df = df[df['node_id'].isin(selected_nodes)].sort_values(['node_id'])
    n_per_sensor = len(df) // num_sensors
    print(f"    每个节点样本数: {n_per_sensor}")

    # 提取 X 列
    x_cols = [f'x_t-{input_steps-1-i}_d{j}' for i in range(input_steps) for j in range(2)]
    y_cols = [f'y_t+{i+1}_d{j}' for i in range(output_steps) for j in range(2)]

    # 转为 numpy 并重塑
    x_data = df[x_cols].values.astype(np.float32)  # (S*N, 24)
    y_data = df[y_cols].values.astype(np.float32)  # (S*N, 24)

    # 重塑为 (S, N, T, F)
    x_data = x_data.reshape(num_sensors, n_per_sensor, input_steps, 2)
    y_data = y_data.reshape(num_sensors, n_per_sensor, output_steps, 2)

    print(f"    X shape: {x_data.shape}, Y shape: {y_data.shape}")

    np.save(out_x, x_data)
    np.save(out_y, y_data)
    print(f"    已保存: {out_x}, {out_y}")
    del df, x_data, y_data


# ==================== 预处理 ====================
print("=" * 50)
print("PEMS-BAY 本地预处理（监督学习格式）")
print("=" * 50)

print("\n[1/3] train.parquet")
parquet_to_xy(train_parquet, train_out_x, train_out_y, NUM_SENSORS)

print("\n[2/3] val.parquet")
parquet_to_xy(val_parquet, val_out_x, val_out_y, NUM_SENSORS)

print("\n[3/3] test.parquet")
parquet_to_xy(test_parquet, test_out_x, test_out_y, NUM_SENSORS)

print("\n[4/4] adj_mx.npy")
adj = np.load(adj_npy)
print(f"    形状: {adj.shape}")
adj_50 = adj[:NUM_SENSORS, :NUM_SENSORS].astype(np.float32)
np.save(adj_out, adj_50)
print(f"    已保存: {adj_out}, 形状: {adj_50.shape}")

print("\n" + "=" * 50)
print("预处理完成！输出文件：")
for f in [train_out_x, train_out_y, val_out_x, val_out_y, test_out_x, test_out_y, adj_out]:
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / 1024 / 1024
        print(f"  {os.path.basename(f)}: {size_mb:.1f} MB")
print("=" * 50)
