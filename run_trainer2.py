import pickle

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from HousePriceProcessor import HousePriceProcessor


def main():
    # 配置生产级参数
    CONFIG = {
        "numerical": ['面积', '房龄'],
        "categorical": ['户型', '装修等级', '近地铁', '朝向', '区域']
    }

    try:
        # 1. 数据加载与校验
        raw_df = pd.read_excel("data/raw/sh_house.xlsx")
        print(f"原始数据维度: {raw_df.shape}")

        # 2. 初始化工业级处理器
        processor = HousePriceProcessor(CONFIG)

        # 3. 执行生产级数据处理
        X, y = processor.process_data(raw_df)
        print(f"预处理后数据维度: {X.shape}")

        # 4. 数据集划分（生产级随机种子）
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )

        # 5. 训练生产级模型
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 6. 模型评估（专业级指标）
        y_pred = model.predict(X_test)
        print(f"\n模型性能报告:")
        print(f"R²分数: {r2_score(y_test, y_pred):.4f}")
        print(f"系数数量: {len(model.coef_)}")

        # 7. 保存生产级资产
        asset = {
            'model': model,
            'preprocessor': processor.preprocessor,
            'feature_names': processor.feature_names,
            'config': CONFIG
        }
        joblib.dump(asset, "models/production_model.pkl")
        joblib.dump(processor,"models/production_preprocessor.pkl")
        print("\n生产级模型保存成功")

    except Exception as e:
        print(f"\n生产流程异常: {str(e)}")
        raise


if __name__ == "__main__":
    main()