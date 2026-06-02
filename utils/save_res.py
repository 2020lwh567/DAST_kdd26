import pandas as pd

def save_metrics_to_excel(best_metrics, excel_path):
    # 训练结束后,将最佳指标结果保存到Excel文件
    df = pd.DataFrame([best_metrics])  # 将指标结果转换成DataFrame
    try:
        # 尝试打开现有的Excel文件来追加数据
        with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            original_df = pd.read_excel(excel_path)     # 如果文件已存在，读取原始数据
            df = pd.concat([original_df, df], ignore_index=True)        # 将新数据追加到原始数据
            df.to_excel(writer, index=False)
    except FileNotFoundError:   # 如果文件不存在，创建一个新文件
        with pd.ExcelWriter(excel_path, mode='w', engine='openpyxl') as writer:
            df.to_excel(writer, index=False)