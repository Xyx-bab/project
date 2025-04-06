# backend/run_trainer.py
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score


# 专业级数据验证装饰器
def validate_data_shape(func):
    def wrapper(*args,** kwargs):
        result = func(*args, ** kwargs)
        X, y = result
        assert not np.isnan(X).any(), "存在未处理的缺失值"
        assert X.shape[1] == len(args[0].feature_names), \
            f"特征数量不匹配: 数据{X.shape[1]} vs 名称{len(args[0].feature_names)}"
        return result

    return wrapper


class HousePriceProcessor:
    def __init__(self, config):
        self.config = config
        self.feature_names = []
        self._build_preprocessor()

    def _build_preprocessor(self):
        """构建工业级预处理管道"""
        numerical_features = self.config['numerical']
        categorical_features = self.config['categorical']

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numerical_features),
                ('onehot', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(
                        handle_unknown='ignore',
                        sparse_output=False
                    ))
                ]), categorical_features)
            ],
            remainder='drop'
        )

    def _generate_feature_names(self, df):
        """动态生成专业级特征名称"""
        num_features = self.config['numerical']

        # 获取编码后的分类特征名称
        onehot_pipe = self.preprocessor.named_transformers_['onehot']
        encoder = onehot_pipe.named_steps['encoder']
        cat_features = encoder.get_feature_names_out(self.config['categorical'])

        self.feature_names = num_features + cat_features.tolist()
        print(f"生成专业级特征 {len(self.feature_names)} 个")

    @validate_data_shape
    def process_data(self, df):
        """执行生产级数据处理"""
        X = df.drop("总价", axis=1)
        y = df["总价"].values

        # 执行预处理
        X_processed = self.preprocessor.fit_transform(X)
        self._generate_feature_names(df)

        return X_processed, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """生产级转换方法"""
        processed = self.preprocessor.transform(df)
        return pd.DataFrame(processed, columns=self.feature_names)
