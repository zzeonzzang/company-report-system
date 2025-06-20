#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Revenue Prediction Model
Based on teammate's [노창호] quarterly revenue prediction model
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import sys
import json
import warnings
warnings.filterwarnings("ignore")

def predict_next_quarter_revenue(quarterly_data):
    """
    Predict next quarter revenue using RandomForest model
    
    Args:
        quarterly_data: List of dictionaries with quarterly financial data
        Format: [{"year": 2024, "quarter": "Q1", "revenue": "xxx", "operatingProfit": "xxx", ...}]
    
    Returns:
        Dictionary with prediction results
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(quarterly_data)
        
        # Ensure we have enough data
        if len(df) < 2:
            return {
                "error": "최소 2분기 데이터가 필요합니다",
                "predictedRevenue": None,
                "accuracy": 0,
                "confidence": "low"
            }
        
        # Sort by year and quarter
        quarter_order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
        df['quarter_num'] = df['quarter'].map(quarter_order)
        df = df.sort_values(['year', 'quarter_num']).reset_index(drop=True)
        
        # Convert financial data to numeric (조 단위)
        financial_columns = ['revenue', 'operatingProfit', 'netIncome', 
                           'totalAssets', 'totalEquity', 'currentAssets', 'currentLiabilities']
        
        for col in financial_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce') / 1e12
        
        # Q4 매출액 누계 보정 (Q4 = 연간누계 - Q1 - Q2 - Q3)
        df_corrected = df.copy()
        for year in df['year'].unique():
            year_data = df[df['year'] == year]
            q4_data = year_data[year_data['quarter'] == 'Q4']
            
            if not q4_data.empty and len(year_data) == 4:
                q1_rev = year_data[year_data['quarter'] == 'Q1']['revenue'].values[0] if not year_data[year_data['quarter'] == 'Q1'].empty else 0
                q2_rev = year_data[year_data['quarter'] == 'Q2']['revenue'].values[0] if not year_data[year_data['quarter'] == 'Q2'].empty else 0
                q3_rev = year_data[year_data['quarter'] == 'Q3']['revenue'].values[0] if not year_data[year_data['quarter'] == 'Q3'].empty else 0
                q4_idx = q4_data.index[0]
                
                # Q4 단일 분기 매출 = 연간 누계 - (Q1 + Q2 + Q3)
                q4_annual = df_corrected.loc[q4_idx, 'revenue']
                q4_single = q4_annual - (q1_rev + q2_rev + q3_rev)
                df_corrected.loc[q4_idx, 'revenue'] = max(q4_single, 0)  # 음수 방지
        
        df = df_corrected.copy()
        
        # Feature engineering
        features = []
        
        # 기본 재무 지표
        if 'revenue' in df.columns:
            features.append('revenue')
        if 'operatingProfit' in df.columns:
            features.append('operatingProfit')
        if 'netIncome' in df.columns:
            features.append('netIncome')
        if 'totalAssets' in df.columns:
            features.append('totalAssets')
        if 'totalEquity' in df.columns:
            features.append('totalEquity')
        
        # 추가 비율 지표 생성
        if 'operatingProfit' in df.columns and 'revenue' in df.columns:
            df['operating_margin'] = df['operatingProfit'] / (df['revenue'] + 1e-10)
            features.append('operating_margin')
        
        if 'netIncome' in df.columns and 'revenue' in df.columns:
            df['net_margin'] = df['netIncome'] / (df['revenue'] + 1e-10)
            features.append('net_margin')
        
        if 'totalEquity' in df.columns and 'totalAssets' in df.columns:
            df['equity_ratio'] = df['totalEquity'] / (df['totalAssets'] + 1e-10)
            features.append('equity_ratio')
        
        # 성장률 지표
        if len(df) >= 2:
            df['revenue_growth'] = df['revenue'].pct_change()
            features.append('revenue_growth')
        
        # Target variable: 다음 분기 매출액
        df['target'] = df['revenue'].shift(-1)
        
        # Prepare data for training
        X = df[features].fillna(0)
        y = df['target']
        
        # Remove last row (no target) and rows with missing targets
        valid_mask = y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 2:
            return {
                "error": "학습용 데이터가 부족합니다",
                "predictedRevenue": None,
                "accuracy": 0,
                "confidence": "low"
            }
        
        # Split data (시계열이므로 shuffle=False)
        if len(X) >= 4:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, shuffle=False
            )
        else:
            # 데이터가 적으면 전체를 학습에 사용
            X_train, y_train = X, y
            X_test, y_test = X.iloc[[-1]], y.iloc[[-1]]
        
        # Train RandomForest model (teammate's optimal parameters)
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        if len(X_test) > 0:
            y_pred_test = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            accuracy = max(0, 100 - (mae / y_test.mean() * 100))
        else:
            accuracy = 75  # Default accuracy
        
        # Predict next quarter
        latest_features = X.iloc[[-1]]
        next_quarter_prediction = model.predict(latest_features)[0]
        
        # Convert back to original scale (조 → 원)
        predicted_revenue_won = next_quarter_prediction * 1e12
        
        # Calculate confidence based on data quality and consistency
        confidence = "high" if len(df) >= 6 and accuracy > 80 else "medium" if len(df) >= 4 else "low"
        
        # Feature importance
        feature_importance = dict(zip(features, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "predictedRevenue": int(predicted_revenue_won),
            "accuracy": round(accuracy, 2),
            "confidence": confidence,
            "model": "RandomForest",
            "features_used": len(features),
            "data_points": len(df),
            "top_features": [{"name": k, "importance": round(v*100, 2)} for k, v in top_features],
            "mae": round(mae * 1e12, 0) if 'mae' in locals() else None
        }
        
    except Exception as e:
        return {
            "error": f"예측 중 오류 발생: {str(e)}",
            "predictedRevenue": None,
            "accuracy": 0,
            "confidence": "low"
        }

if __name__ == "__main__":
    # Read input from stdin (Node.js will pass data this way)
    if len(sys.argv) > 1:
        # Command line argument
        input_data = sys.argv[1]
    else:
        # Stdin
        input_data = sys.stdin.read()
    
    try:
        quarterly_data = json.loads(input_data)
        result = predict_next_quarter_revenue(quarterly_data)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as e:
        error_result = {
            "error": f"입력 데이터 처리 오류: {str(e)}",
            "predictedRevenue": None,
            "accuracy": 0,
            "confidence": "low"
        }
        print(json.dumps(error_result, ensure_ascii=False))