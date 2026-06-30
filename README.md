# Person Detection and Full-Body Tracking System

Sistem deteksi dan pelacakan manusia (person detection & tracking) multi-stream menggunakan algoritma YOLO (Ultralytics) dan OpenCV. Sistem ini dirancang untuk membaca input dari beberapa kamera (RTSP stream atau webcam), mendeteksi objek manusia secara real-time, mengestimasi posisi 3 dimensi (3D) relatif terhadap kamera menggunakan kamera tunggal (monocular depth estimation), melacak pergerakan objek dengan filter penghalus bounding box (smoothing), serta mengekspor status deteksi secara thread-safe ke dalam file JSON secara periodik.

---

## 🚀 Fitur Utama

1. **Multi-Stream Processing**: Memproses beberapa input video atau RTSP stream secara paralel menggunakan multi-threading (`threading.Thread`).
2. **Real-time Object Detection**: Menggunakan model YOLO dari Ultralytics (default: `yolo26n.pt` atau model YOLOv8/v11 lainnya) untuk deteksi objek manusia (Class ID: 0).
3. **Monocular 3D Position Estimation**: Mengestimasi jarak/kedalaman ($z$) dan posisi lateral ($x, y$) manusia dalam meter menggunakan informasi dimensi bounding box dari kamera tunggal.
4. **Advanced Object Tracking & Smoothing**:
   - Mengasosiasikan objek antar-frame menggunakan Euclidean Distance.
   - Mengurangi getaran (jitter) visual pada bounding box menggunakan Exponential Moving Average (EMA) smoothing.
   - Membatasi perubahan ukuran bounding box yang drastis (`BOX_MAX_SIZE_CHANGE_RATIO`).
   - Fitur **Stationary Size Lock** untuk mengunci ukuran bounding box ketika posisi target dianggap diam/stasioner, mencegah fluktuasi ukuran akibat noise deteksi.
5. **Thread-Safe State Management**: Mengagregasikan data deteksi dari seluruh stream dan menyimpannya secara periodik ke dalam file `current_state.json` hanya jika terdapat perubahan data (state changes).
6. **Robust Logging System**: Logging terpusat dengan rotasi file otomatis (max 2MB, 5 backup) yang mencatat aktivitas sistem dan mempermudah debugging.

---

## 🛠️ Persyaratan Sistem & Instalasi

### 1. Prasyarat
Pastikan Anda sudah menginstal Python (versi 3.8 ke atas direkomendasikan).

### 2. Instalasi Dependensi
Instal pustaka Python yang diperlukan dengan menjalankan perintah berikut di terminal:
```bash
pip install -r requirements.txt
```
*Isi `requirements.txt`:*
* `ultralytics` (untuk YOLO)
* `opencv-python` (untuk pemrosesan video dan visualisasi)
* `numpy` (untuk komputasi matriks/jarak)
* `python-dotenv` (untuk konfigurasi berbasis file `.env`)

---

## ⚙️ Konfigurasi (File `.env`)

Sistem menggunakan konfigurasi dinamis melalui variabel lingkungan yang dapat didefinisikan dalam file `.env` di direktori utama proyek. Buat file `.env` dan sesuaikan parameter berikut:

| Parameter | Default | Deskripsi |
| :--- | :--- | :--- |
| `MODEL_PATH` | `yolo26n.pt` | Jalur (path) ke file model YOLO (misal: `yolov8n.pt`). |
| `RTSP_URLS` | `0` | Sumber video. Gunakan indeks angka (misal `0`) untuk webcam, atau alamat URL RTSP dipisahkan koma untuk multi-stream (misal: `rtsp://user:pass@ip:port/h264,rtsp://user:pass@ip2:port/h264`). |
| `LOG_PATH` | `logs/detections.log`| Jalur penyimpanan file log. |
| `DETECT_CLASS` | `0` | ID kelas objek yang dideteksi (0 untuk manusia/person di dataset COCO). |
| `MIN_YOLO_CONFIDENCE` | `0.75` | Batas minimal kepercayaan (confidence threshold) deteksi YOLO. |
| `PERSON_NMS_IOU_THRESHOLD`| `0.45` | Ambang batas NMS (IoU) untuk menyaring bounding box yang tumpang tindih. |
| `PERSON_REAL_HEIGHT_M` | `1.70` | Estimasi tinggi rata-rata manusia dalam meter untuk kalkulasi kedalaman (z). |
| `CAMERA_FOCAL_LENGTH_PX` | `700.0` | Panjang fokus kamera dalam piksel (perlu dikalibrasi untuk akurasi jarak). |
| `FRAME_SKIP` | `0` | Jumlah frame yang dilewati per pemrosesan YOLO untuk meningkatkan performa (0 = proses semua frame). |
| `TRACKER_MAX_UNSEEN` | `10` | Batas maksimum frame objek tidak terdeteksi sebelum pelacak (tracker) dihapus. |
| `PERSON_TRACKER_MAX_DISTANCE`| `75` | Jarak piksel maksimum untuk mengasosiasikan deteksi baru dengan pelacak yang ada. |
| `BOX_SMOOTHING_ALPHA` | `0.70` | Koefisien smoothing (0.0 - 1.0). Nilai tinggi = lebih mulus/lambat merespon perubahan; Nilai rendah = lebih responsif/sensitif jitter. |
| `BOX_MAX_SIZE_CHANGE_RATIO` | `0.20` | Batas maksimum perubahan ukuran bounding box antar frame (persentase). |
| `BOX_STATIONARY_CENTER_THRESHOLD` | `8.0` | Toleransi pergeseran pusat box (dalam piksel) untuk mengaktifkan penguncian ukuran stasioner. |
| `STATE_FILE_PATH` | `current_state.json` | Jalur penyimpanan output status deteksi dalam format JSON. |
| `HISTORY_FILE_PATH` | `state_history.jsonl`| Jalur untuk menyimpan riwayat status (opsional). |
| `STATE_UPDATE_INTERVAL_SECONDS`| `5` | Interval waktu (detik) untuk memperbarui dan menulis status ke file JSON. |

---

## 🖥️ Cara Menjalankan Program

Jalankan skrip utama `main.py` menggunakan Python:
```bash
python main.py
```

### Navigasi Selama Program Berjalan:
* Tekan tombol **'q'** pada jendela OpenCV untuk menghentikan program secara aman (graceful shutdown). Program akan menunggu seluruh thread stream selesai sebelum keluar sepenuhnya.

---

## 📊 Output Data (`current_state.json`)

Status deteksi real-time disimpan secara periodik ke dalam file JSON dengan struktur berikut:
```json
{
    "timestamp": "2026-06-30T10:30:00.123456",
    "streams": {
        "0": {
            "persons": [
                {
                    "person_id": 0,
                    "class_id": 0,
                    "confidence": 0.8952,
                    "bbox": {
                        "x1": 120,
                        "y1": 80,
                        "x2": 240,
                        "y2": 320
                    },
                    "position": {
                        "image": {
                            "x": 180.0,
                            "y": 200.0,
                            "bbox_width": 120,
                            "bbox_height": 240
                        },
                        "camera": {
                            "x": -0.121,
                            "y": 0.081,
                            "z": 4.958
                        }
                    }
                }
            ]
        }
    }
}
```

---

## 📐 Penjelasan Rumus Estimasi Posisi 3D (Monocular)

Untuk mengestimasi posisi koordinat 3D $(x, y, z)$ dalam meter relatif terhadap sensor kamera dari gambar 2D, sistem ini memanfaatkan proyeksi kamera lubang jarum (*pinhole camera model*):

1. **Jarak / Kedalaman ($z$):**
   $$z = \frac{H_{\text{real}} \times f}{h_{\text{bbox}}}$$
   * $H_{\text{real}}$: Estimasi tinggi nyata objek manusia dalam meter (default: `1.70` m).
   * $f$: Panjang fokus kamera dalam piksel (`CAMERA_FOCAL_LENGTH_PX`).
   * $h_{\text{bbox}}$: Tinggi bounding box objek dalam piksel.

2. **Posisi Horisontal ($x$):**
   $$x = \frac{(X_{\text{center}} - X_{\text{principal}}) \times z}{f}$$
   * $X_{\text{center}}$: Koordinat pusat horisontal bounding box pada gambar.
   * $X_{\text{principal}}$: Titik pusat horisontal gambar (lebar gambar / 2).

3. **Posisi Vertikal ($y$):**
   $$y = \frac{(Y_{\text{center}} - Y_{\text{principal}}) \times z}{f}$$
   * $Y_{\text{center}}$: Koordinat pusat vertikal bounding box pada gambar.
   * $Y_{\text{principal}}$: Titik pusat vertikal gambar (tinggi gambar / 2).

---

## 📁 Struktur Berkas Proyek

* `main.py`: Skrip utama yang mengatur inisialisasi model, multi-threading pemrosesan stream video, pelacakan koordinat, penggambaran visualisasi OpenCV, serta loop pembaruan status.
* `config.py`: File pembaca konfigurasi berbasis `dotenv` yang memuat parameter deteksi, kamera, pelacakan, dan lokasi output.
* `state_manager.py`: Modul thread-safe untuk mengelola agregasi data koordinat aktif dari semua kamera dan menulisnya ke file JSON jika terjadi perubahan.
* `logger_config.py`: Konfigurasi logging aplikasi dengan format debug mendetail dan rotasi file otomatis.
* `requirements.txt`: Daftar pustaka Python yang wajib diinstal sebelum menjalankan aplikasi.
* `.gitignore`: File pengabaian git untuk mencegah file log, model, dan status terunggah ke repositori.
