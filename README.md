
# Klasifikasi Kualitas Teh Berdasarkan Aroma Sensor

Proyek ini bertujuan untuk mengklasifikasikan jenis teh berdasarkan sinyal aroma yang direkam oleh enam sensor gas. Data dari setiap sampel teh dianalisis menggunakan ekstraksi fitur statistik dan model machine learning untuk memprediksi kualitas/jenis teh.

---

## Dataset

Dataset berasal dari [data_teh](https://github.com/wicaksonoleksono/data_teh). Tiap file `.csv` mewakili satu rekaman sensor dari satu sampel teh dengan ukuran 3000 baris × 6 kolom:

- MQ 7
- MQ 9
- TGS 822
- TGS 2600
- TGS 2602
- TGS 2611

Data dibagi menjadi:
- `data_teh_split/train/` → data latih
- `data_teh_split/test/` → data uji

---

## Struktur Proyek

```
.
├── data_teh/                    # Data mentah sensor
├── data_teh_split/              # Data train/test hasil split
│   ├── train/
│   └── test/
├── train_dataset.csv            # Data latih hasil ekstraksi fitur
├── test_dataset.csv             # Data uji hasil ekstraksi fitur
├── output/
│   └── model_teh.pkl            # Model Random Forest
│   └── model_xgb_teh.pkl        # Model XGBoost
├── main.py                      # Proses split dan ekstraksi fitur
├── train_model.py               # Latih & evaluasi model XGBoost
├── predict_teh.py               # Prediksi file tunggal
├── src/
│   ├── preprocess.py
│   └── build_dataset.py
```

---

## Cara Menjalankan

1. **Install library**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib
   ```

2. **Preprocessing & split data**
   ```bash
   python main.py
   ```

3. **Latih model**
   - Random Forest:
     ```bash
     python train_model.py
     ```
   - XGBoost:
     ```bash
     python train_model_xgboost.py
     ```

4. **Prediksi file tunggal**
   ```bash
   python predict_teh.py --file data_teh/F1/F1_6.csv
   ```

---

## Fitur yang Diekstraksi

Dari setiap sensor, diambil 6 fitur statistik:
- Mean
- Standard Deviation
- Max
- Min
- Skewness
- Kurtosis

Total: **6 sensor × 6 fitur = 36 fitur per sampel**

---

## Evaluasi Model

Evaluasi dilakukan dengan:
- Classification Report (Precision, Recall, F1)
- Confusion Matrix
- Akurasi data uji

---

## Pengembangan Selanjutnya

- Deploy model sebagai aplikasi web
- Ekstensi ke LSTM untuk data sekuens mentah

---

## 👤 Pembuat

Dikembangkan oleh [@weyennn](https://github.com/weyennn) dengan ☕ dan data.
