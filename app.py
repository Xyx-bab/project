from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS
import pandas as pd
import os
import joblib
from flask import Flask, jsonify
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression


app = Flask(__name__,
    template_folder=os.path.join('frontend/templates'),
    static_folder=os.path.join('frontend/static')

)
CORS(app)
# 文件路径配置
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_PATH = os.path.join('models/xgboost_model.pkl')
users_file = os.path.join('data', 'users.xlsx')

# 加载房价预测模型
model = joblib.load(MODEL_PATH)


@app.route('/')
def index():
    return send_from_directory('frontend/templates', 'login.html')

@app.route('/chart1')
def chart1():
    return send_from_directory('frontend/templates', 'showchart.html')

@app.route('/register1')
def register1():
    return send_from_directory('frontend/templates', 'register.html')

@app.route('/predict1')
def predict1():
    return send_from_directory('frontend/templates', 'predict.html')
# 用户认证接口
@app.route('/api/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')
    datapath = os.path.join('data', 'users.xlsx')
    users = pd.read_excel(
        datapath,
        engine='openpyxl',
        dtype={'username': str, 'password': str})
    user = users[(users['username'] == username) & (users['password'] == password)]

    if not user.empty:
        return jsonify({'success': True, 'user': user.iloc[0].to_dict()})
    return jsonify({'success': False, 'message': '认证失败'}), 401


# 房价数据接口
@app.route('/api/house-data')
def get_house_data():
    data = pd.read_csv(os.path.join(DATA_DIR, 'house_data.csv'))
    return jsonify(data.to_dict(orient='records'))


@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(data)
        model = joblib.load('models/xgboost_model.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')

        # 必填字段验证
        required_fields = [
            '区域', '房龄', '户型',
            '装修等级', '区域', '朝向', '近地铁'
        ]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"缺少必要字段: {field}"}), 400

        # 数值验证
        validation_rules = [
            (data['面积'] > 0, "面积必须大于0"),
            (data['房龄'] >= 0, "房龄不能为负数"),
            (data['户型'] in ['1室1厅', '2室1厅', '3室1厅'], "无效户型"),
            (data['装修等级'] in ['简装', '精装', '毛坯'], "无效装修等级"),
            (data['区域'] in ['浦东', '黄浦', '徐汇', '静安', '长宁'], "无效区域"),
            (data['朝向'] in ['西', '南', '东', '北'], "无效朝向")
        ]

        print("验证成功")

        for condition, message in validation_rules:
            if not condition:
                return jsonify({"error": message}), 400

        # 执行预测
        input_data = {
            '面积': float(data['面积']),
            '房龄': int(data['房龄']),
            '户型': data['户型'],
            '装修等级': data['装修等级'],
            '区域': data['区域'],
            '朝向': data['朝向'],
            '近地铁': data['近地铁']
        }
        X= pd.DataFrame([input_data])
        X_processed = preprocessor.transform(X)

        # 预测价格
        result = model.predict(X_processed)[0]


        print(result)



        return jsonify({
            "success": True,
            "predicted_price": float(result),
            "currency": "万元"
        })

    except Exception as e:
        app.logger.error(f"预测失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"预测服务异常: {str(e)}"
        }), 500


@app.route('/api/register', methods=['POST'])
def register():
    try:
        # 读取现有数据
        if os.path.exists(users_file):
            df = pd.read_excel(users_file)
        else:
            df = pd.DataFrame(columns=['username', 'password'])

        # 获取注册数据
        data = request.json
        username = data['username']
        password = data['password']

        # 验证用户名是否存在
        if username in df['username'].values:
            return jsonify({"success": False, "message": "用户名已存在"}), 400

        # 追加新用户
        new_user = pd.DataFrame([[username, password]], columns=['username', 'password'])
        df = pd.concat([df, new_user], ignore_index=True)

        # 保存数据
        df.to_excel(users_file, index=False)
        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


districts = ['浦东', '闵行', '宝山', '徐汇', '普陀', '杨浦', '长宁', '松江', '嘉定', '黄浦']


def process_data():
    """数据处理与预测"""
    df = pd.read_excel('data/raw/sh_house_data.xlsx')
    df = df.set_index('区域').T
    df.index = pd.to_datetime(df.index)

    predictions = {}
    for district in districts:
        model = LinearRegression()
        X = np.arange(len(df)).reshape(-1, 1)
        y = df[district].values.reshape(-1, 1)
        model.fit(X, y)
        next_month = model.predict([[len(df)]])[0][0]
        predictions[district] = int(next_month)

    return df, predictions


@app.route('/api/data')
def get_all_data():
    """获取全部数据"""
    df, _ = process_data()
    return jsonify({
        'dates': df.index.strftime('%Y-%m').tolist(),
        'data': {d: df[d].tolist() for d in districts}
    })


@app.route('/api/predict/<district>')
def get_district_data(district):
    """获取单个区域数据"""
    df, predictions = process_data()
    return jsonify({
        'history': {
            'dates': df.index.strftime('%Y-%m').tolist(),
            'prices': df[district].tolist()
        },
        'prediction': predictions[district]
    })


if __name__ == '__main__':
    app.run(port=5000, debug=True)
