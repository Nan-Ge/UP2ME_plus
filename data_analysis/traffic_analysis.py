from dask.distributed import Client, LocalCluster
import dask.dataframe as dd
import pandas as pd
import numpy as np


if __name__ == "__main__":
    # Dask paras
    n_workers = 8  # Number of workers
    threads_per_worker = 2  # Number of threads per worker

    # 创建一个本地集群，指定工作进程数和每个进程的线程数
    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker)  # 启动 4 个工作进程，每个进程使用 2 个线程
    client = Client(cluster)

    # 打印集群信息
    print(client)

    # 加载 CSV 文件和处理数据
    df = dd.read_csv('./datasets/vd_traffic/raw_dataset/metric/compute/part_*.csv').repartition(npartitions=32)

    # 打印结果
    print(df.head(100))

    # 关闭客户端和集群（可选，调试完成后关闭）
    client.close()
    cluster.close()
