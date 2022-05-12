################################################################
# 타이타닉 생존 예측 웹 서비스
################################################################
# -------------------------------------------------------------
# 1. 라이브러리 로딩
from flask_restful import reqparse 
from flask import Flask
import numpy as np 
import pandas as pd
import joblib
import json

# -------------------------------------------------------------
# 2. 앱 선언 및 필요 함수, 변수 선언
# 서버 앱 선언
app = Flask(__name__) 

# 필요한 변수 선언
features = ['INCOME', 'OVERAGE', 'LEFTOVER', 'HOUSE', 'HANDSET_PRICE', 'OVER_15MINS_CALLS_PER_MONTH', 'AVERAGE_CALL_DURATION', 'REPORTED_SATISFACTION', 'REPORTED_USAGE_LEVEL']
target = 'CHURN'
imputer1_list = ['REPORTED_SATISFACTION']
cat = {'REPORTED_SATISFACTION':["very_unsat", "very_sat", "unsat", "avg", "sat"],
       'REPORTED_USAGE_LEVEL':["little", "very_high", "very_little", "high", "avg"]}

# 필요한 함수 선언
def mobile_dumm(df, cat):
    temp = df.copy()
    for k, v in cat.items():
        temp[k] = pd.Categorical(temp[k], categories=v, ordered=False)

    temp = pd.get_dummies(temp, columns=cat.keys(), drop_first=True)
    return temp

def mobile_datapipeline(df, simpleimputer, simple_impute_list, dumm_list, scaler, knnimputer):

    temp = df.copy()

    # Feature Engineering
#     temp = mobile_fe(temp)

    # NaN 조치① : SimpleImputer
    temp[simple_impute_list] = simpleimputer.fit_transform(temp[simple_impute_list])

    # 가변수화
    temp = mobile_dumm(temp, dumm_list)

    x_cols = list(temp)
    # 스케일링
    temp = scaler.transform(temp)

    # NaN 조치② : KNNImputer
    temp = knnimputer.transform(temp)

    return pd.DataFrame(temp, columns = x_cols)
# -------------------------------------------------------------
# 3. 웹서비스

@app.route('/predict/', methods=['POST']) 
def predict(): 

    # 입력받은 json 파일에서 정보 뽑기(파싱)
    parser = reqparse.RequestParser() 
    for v in features :
        parser.add_argument(v, action='append') 

    # 뽑은 값을 딕셔너리로 저장
    args = parser.parse_args() 
        
    # 딕셔너리를 데이터프레임(2차원으로 만들기)
    x_input = pd.DataFrame(args)

    # 전처리
    x_input = mobile_datapipeline(x_input, imputer1, imputer1_list, cat, scaler, imputer2)

    # 예측. 결과는 넘파이 어레이
    pred = model.predict(x_input) 

    # 결과를 숫자로 반환
    result1 = [int(val) for val in list(pred)]

    # 결과를 식별가능한 문자로 변환(0,1로 반환할 때, 타입오류가 날 수 있음.)
    result2 = np.where(pred == 0, 'Stay','Leave')
    
    # result : json 형태로 전송해야 한다.
    out = {'pred1': result1, 'pred2': list(result2)} 
   
    return out

@app.route("/")
def index():
    
    
# -------------------------------------------------------------
# 4.웹서비스 직접 실행시 수행 
if __name__ == '__main__': 

    # 전처리 + 모델 불러오기
    imputer1 = joblib.load('preprocess/imputer1_mo.pkl')
    imputer2 = joblib.load('preprocess/imputer2_mo.pkl')
    scaler = joblib.load( 'preprocess/scaler_mo.pkl')
    model = joblib.load('model/model_mo.pkl')
    


    # 웹서비스 실행 및 접속 정보
    app.run(host='127.0.0.1', port=8080, debug=True)

# -------------------------------------------------------------
