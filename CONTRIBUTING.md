
# Panduan Kontribusi – Klasifikasi Kualitas Teh

---

## Cara Berkontribusi

1. **Fork repository ini**
   - Klik tombol "Fork" di kanan atas halaman repo

2. **Clone repo hasil fork ke komputer kamu**
   ```bash
   git clone https://github.com/weyennn/tea-classifier.git
   cd tea-classifier
   ```

3. **Buat branch baru untuk fitur atau perbaikan**
   ```bash
   git checkout -b nama-fitur
   ```

4. **Lakukan perubahan dan commit dengan pesan jelas**
   ```bash
   git add .
   git commit -m "feat: tambah modul prediksi batch"
   ```

5. **Push branch ke GitHub**
   ```bash
   git push origin nama-fitur
   ```

6. **Buat Pull Request**
   - Jelaskan perubahan yang dilakukan dan alasan kamu menambahkannya

---

## Style Panduan

- Gunakan Pythonic code dan clean structure
- Simpan model/output di folder `output/`

---

## Testing

Pastikan semua script berjalan normal:
- `main.py` untuk preprocessing & split
- `train_model.py` berjalan tanpa error
- Prediksi bisa dilakukan melalui `predict.py`

---


---

Tks.