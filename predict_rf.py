import pandas as pd
import joblib

rf_model = joblib.load("rf_model.pkl")
xgb_model=joblib.load("xgb_model.pkl")
print("✅ Model loaded.")

print("\n--- Hypertension Prediction ---")
print("Type 'quit' at any time to exit.\n")

def get_input(prompt, dtype=float, min_val=None, max_val=None):
    """Safe input function with validation and ranges"""
    while True:
        try:
            val = dtype(input(f"{prompt} "))
            if min_val is not None and val < min_val:
                print(f"⚠️ Value must be ≥ {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"⚠️ Value must be ≤ {max_val}")
                continue
            return val
        except ValueError:
            print("⚠️ Invalid input. Try again.")

while True:
   
    choice = input("Do you want to enter a new patient? (yes/quit): ").strip().lower()
    if choice == "quit":
        print("👋 Exiting prediction loop.")
        break
    if choice != "yes":
        continue

    try:
       
        hrecord = get_input("HRecord (hours, 15.5-24):", float, 15.5, 24)
        perc      = get_input("Perc (%):", float, 0, 100)
        interrupt = get_input("Interrupt (0/1):", int, 0, 1)
        age       = get_input("Age:", int, 0, 120)
        weight    = get_input("Weight (kg):", float, 20, 200)
        height    = get_input("Height (cm):", float, 50, 250)
        bps_day   = get_input("BPS-Day24 (mmHg):", float, 80, 250)
        bpd_day   = get_input("BPD-Day24 (mmHg):", float, 40, 150)
        bps_night = get_input("BPS-Night24 (mmHg):", float, 70, 220)
        bpd_night = get_input("BPD-Night24 (mmHg):", float, 40, 140)
        sex       = get_input("Sex (0=Female, 1=Male):", int, 0, 1)
        bp_load   = get_input("BP-Load (%):", float, 0, 100)
        max_sys   = get_input("Max-Sys (mmHg):", float, 80, 260)
        min_sys   = get_input("Min-Sys (mmHg):", float, 50, 200)
        max_dia   = get_input("Max-Dia (mmHg):", float, 40, 160)
        min_dia   = get_input("Min-Dia (mmHg):", float, 30, 120)

        
        new_data = pd.DataFrame([{
            "HRecord": hrecord,
            "Perc": perc,
            "Interrupt": interrupt,
            "Age": age,
            "Weight": weight,
            "Height": height,
            "BPS-Day24": bps_day,
            "BPD-Day24": bpd_day,
            "BPS-Night24": bps_night,
            "BPD-Night24": bpd_night,
            "Sexe": sex,       # keep same column name as training
            "BP-Load": bp_load,
            "Max-Sys": max_sys,
            "Min-Sys": min_sys,
            "Max-Dia": max_dia,
            "Min-Dia": min_dia
        }])

    
        prediction = rf_model.predict(new_data)[0]
        probabilities = rf_model.predict_proba(new_data)[0]
        prediction1=xgb_model.predict(new_data)[0]
        probabilities1=xgb_model.predict_proba(new_data)[0]
        print("Random Forest Predictions:\n")
        print("\n✅ Predicted Class:", "Hypertension" if prediction == 1 else "No Hypertension")
        print("🔹 Class Probabilities:", probabilities, "\n")
        print("\n\n========================================\nXGB Predictions:\n") 
        print("\n✅ Predicted Class:", "Hypertension" if prediction1 == 1 else "No Hypertension")
        print("🔹 Class Probabilities:", probabilities1, "\n")
    except Exception as e:
        print(f"⚠️ Error: {e}\n")
