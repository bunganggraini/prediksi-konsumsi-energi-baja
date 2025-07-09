from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("RFModel.pkl")

# Semua fitur yang digunakan saat training
all_features = [
    'Hour', 'Day', 'Month', 'NSM',
    'Lagging_Current_Reactive.Power_kVarh',
    'Leading_Current_Reactive_Power_kVarh',
    'CO2(tCO2)',
    'Lagging_Current_Power_Factor',
    'Leading_Current_Power_Factor',
    'WeekStatus_Weekend',
    'Day_of_week_Monday', 'Day_of_week_Saturday', 'Day_of_week_Sunday',
    'Day_of_week_Thursday', 'Day_of_week_Tuesday', 'Day_of_week_Wednesday',
    'Load_Type_Maximum_Load', 'Load_Type_Medium_Load'
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        print("📥 DATA MASUK:", data)

        # Siapkan semua kolom dengan nilai default = 0
        row = {col: 0 for col in all_features}

        # Numerik
        for col in [
            'Hour', 'Day', 'Month', 'NSM',
            'Lagging_Current_Reactive.Power_kVarh',
            'Leading_Current_Reactive_Power_kVarh',
            'CO2(tCO2)', 'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor'
        ]:
            try:
                row[col] = float(data.get(col, 0) or 0)
            except:
                row[col] = 0.0

        # WeekStatus
        if data.get("WeekStatus") == "Weekend":
            row["WeekStatus_Weekend"] = 1

        # Day_of_week (Friday = default → semua dummy 0)
        day = data.get("Day_of_week")
        key = f"Day_of_week_{day}"
        if key in row:
            row[key] = 1

        # Load_Type
        load = data.get("Load_Type")
        if load == "Maximum":
            row["Load_Type_Maximum_Load"] = 1
        elif load == "Medium":
            row["Load_Type_Medium_Load"] = 1

        # Buat DataFrame & URUTKAN kolom
        df = pd.DataFrame([row])
        df = df[model.feature_names_in_]

        prediksi = model.predict(df)[0]
        print("✅ Prediksi hasil:", prediksi)
        return jsonify({"prediksi": str(prediksi)})

    except Exception as e:
        print("❌ ERROR SAAT PREDIKSI:", e)
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
