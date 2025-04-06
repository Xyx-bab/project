from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

from services.data_processor import DataProcessor


def main():
    # 加载数据
    df = pd.read_excel("data/raw/sh_house.xlsx")

    # 初始化专业级数据处理器
    processor_config = {
        "numerical": ['面积', '房龄'],
        "categorical": ['户型', '装修等级','近地铁', '朝向', '区域'],
    }
    processor = DataProcessor(processor_config)

    # 数据预处理
    X = processor.fit_transform(df)
    y = df["总价"]

    # 数据集分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 初始化并训练模型
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        tree_method='hist'  # 优化内存使用
    )
    model1= LinearRegression()


    model.fit(X_train, y_train)

    # 保存资产
    joblib.dump(model, "models/model.pkl")
    joblib.dump(processor, "models/preprocessor.pkl")
    print("模型保存成功")


if __name__ == "__main__":
    main()