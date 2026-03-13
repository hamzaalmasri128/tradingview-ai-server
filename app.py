import os
import pandas as pd
from xgboost import XGBClassifier
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss"
)

trained = False
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")


@app.route("/")
def home():
    return "TradingView ML Server Running"


@app.route("/update_data", methods=["POST"])
def update_data():
    global model, trained

    data = request.get_json(silent=True)

    if not data:
        return jsonify({"status": "error", "message": "No JSON data received"}), 400

    if isinstance(data, dict):
        data = [data]

    try:
        df = pd.DataFrame(data)

        required_columns = [
            "bosBull", "liqLow", "fibBuy", "fvgBull",
            "stopHuntBuy", "discount", "bullOB", "scoreHTF"
        ]

        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            return jsonify({
                "status": "error",
                "message": f"Missing required columns: {missing_cols}"
            }), 400

        # تدريب النموذج إذا كانت البيانات تحتوي على trade_result وعدد كافٍ من الصفوف
        if (not trained) and ("trade_result" in df.columns) and (len(df) > 100):
            X = df[required_columns]
            y = df["trade_result"]
            model.fit(X, y)
            trained = True

        # إذا لم يتم تدريب النموذج بعد، نرجع رسالة واضحة بدل حدوث خطأ
        if not trained:
            return jsonify({
                "status": "waiting",
                "message": "Model not trained yet. Send more labeled data with trade_result.",
                "rows_received": len(df)
            }), 200

        # توقع نسبة النجاح
        X_new = df[required_columns]
        df["predicted_success"] = model.predict_proba(X_new)[:, 1] * 100

        payload = df.to_dict(orient="records")

        # إرسال النتائج إلى webhook خارجي إذا كان موجودًا
        if WEBHOOK_URL:
            try:
                requests.post(WEBHOOK_URL, json=payload, timeout=5)
            except requests.RequestException as e:
                return jsonify({
                    "status": "partial_success",
                    "message": "Prediction done, but failed to send webhook",
                    "error": str(e),
                    "data": payload
                }), 200

        return jsonify({
            "status": "ok",
            "trained": trained,
            "data": payload
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
