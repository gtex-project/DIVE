from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Composition

# FastAPI 应用
app = FastAPI()

# =========== 加载模型和特征名 ===========
model = XGBRegressor()
model.load_model('xgb_model.json')

with open('feature_names.txt', 'r') as f:
    feature_names = [line.strip() for line in f.readlines()]

# matminer 特征器
featurizer = ElementProperty.from_preset('magpie', impute_nan=True)

# =========== API 数据模型 ===========
class PredictRequest(BaseModel):
    formula: str

class PredictResponse(BaseModel):
    prediction: float
    formula: str
    markdown: Optional[str] = None

# =========== 辅助函数 ===========
def get_element_fractions(comp):
    if not comp or comp is np.nan:
        return {}
    d = comp.get_el_amt_dict()
    total = sum(d.values())
    return {k: v/total for k, v in d.items()}

# =========== API 路由 ===========
@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        # 1. 化学式处理
        formula = request.formula
        try:
            comp = Composition(formula)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Invalid formula: {formula}"})

        df_new = pd.DataFrame({'composition': [comp]})

        # 2. matminer特征
        X_new = featurizer.featurize_dataframe(df_new, 'composition', ignore_errors=True)

        # 3. 元素摩尔分数特征补全
        element_fractions = get_element_fractions(comp)
        for el in [col.replace('frac_', '') for col in feature_names if col.startswith('frac_')]:
            X_new[f'frac_{el}'] = element_fractions.get(el, 0)

        # 4. 只保留训练时特征列
        try:
            X_new = X_new[feature_names]
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": "Feature mismatch: "+str(e)})

        # 5. 预测
        y_pred_new = model.predict(X_new)
        pred_value = float(y_pred_new[0])

        # 6. 返回
        markdown = f"**Predicted value for `{formula}`:** `{pred_value:.4f}`"
        return PredictResponse(
            prediction=pred_value,
            formula=formula,
            markdown=markdown
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# =========== 你可以加在 app = FastAPI() 后一并注册 ===========
