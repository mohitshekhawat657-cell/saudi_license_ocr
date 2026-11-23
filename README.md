# Saudi Driving License — Authentication System

YOLOv8 detection + VLM OCR (Together AI) + REST API, all in one folder.

---

## Folder structure

```
license_auth/
├── api.py                  ← REST API  (POST /verify, POST /verify-batch)
├── pipeline.py             ← YOLO → crop → OCR → validate
├── ocr_extractor.py        ← VLM OCR engine (Together AI / Transformers)
├── database.py             ← Async PostgreSQL integration (asyncpg)
├── webhook_api.py          ← Webhook delivery + config endpoints
├── bulk_extract.py         ← CLI bulk extraction tool
├── train.py                ← Train YOLO model
├── test_detector.py        ← Test trained model
├── requirements.txt
├── .env.example
├── dataset/
│   └── data.yaml
├── scripts/
│   ├── labelme_to_yolo.py  ← Convert LabelMe annotations → YOLO
│   ├── verify_labels.py    ← Visual check of annotations
│   ├── split_dataset.py    ← Split raw images → train/val/test
│   └── augment.py          ← 5× data augmentation
├── raw_images/             ← Put your license photos here
├── raw_labels/             ← YOLO .txt labels go here (auto-generated)
├── uploads/                ← Temp API uploads (auto-created, auto-cleaned)
└── json_response/          ← Persisted JSON responses (auto-created)
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env — set TOGETHER_API_KEY and TOGETHER_MODEL at minimum
```

System dependency (required by pdf2image for PDF support):
```bash
# macOS
brew install poppler
# Ubuntu
sudo apt install poppler-utils
```

---

## Workflow — train the model first

```bash
# 1. Put license photos in raw_images/
# 2. Annotate with LabelMe
labelme

# 3. Convert annotations to YOLO format
python scripts/labelme_to_yolo.py

# 4. Visually verify annotations
python scripts/verify_labels.py

# 5. Split into train/val/test
python scripts/split_dataset.py

# 6. Augment training data (5×)
python scripts/augment.py

# 7. Train
python train.py

# 8. Test
python test_detector.py
```

---

## Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 3000 --reload
```

Swagger UI → http://localhost:3000/docs

---

## API — POST /verify

| Field        | Type   | Required | Description                                      |
|--------------|--------|----------|--------------------------------------------------|
| `type`       | string | yes      | Document type, e.g. `driving_license`            |
| `driverId`   | string | yes      | Your reference or user ID                        |
| `file`       | image  | yes      | License photo (jpg/png/webp/heic/heif/tiff/bmp/gif/pdf) |
| `webhook_url`| string | no       | Per-request callback URL (overrides WEBHOOK_URL) |

### curl
```bash
curl -X POST http://localhost:3000/verify \
  -F "type=driving_license" \
  -F "driverId=user_123" \
  -F "file=@license.jpg"
```

### Python
```python
import requests

with open("license.jpg", "rb") as f:
    r = requests.post(
        "http://localhost:3000/verify",
        data={"type": "driving_license", "driverId": "user_123"},
        files={"file": ("license.jpg", f, "image/jpeg")},
    )
print(r.json())
```

### Success response `200`
```json
{
  "success": true,
  "type": "driving_license",
  "total": 1,
  "succeeded": 1,
  "failed": 0,
  "index": 0,
  "filename": "license.jpg",
  "detection_confidence": 0.9821,
  "resultData": {
    "license_data": {
      "license_number": "1037582846",
      "full_name_english": "ABDULLAH ALI S ALQARNI",
      "full_name_arabic": "عبدالله بن علي بن سعد القرني",
      "date_of_birth": "20/07/1974",
      "date_of_issue": "17/04/1999",
      "date_of_expiry": "01/04/2033",
      "category": "PRIVATE",
      "nationality": "SAUDI",
      "blood_group": "A+",
      "restrictions": "3",
      "version": "3",
      "issuing_country": "Saudi Arabia"
    },
    "isValid": true,
    "sequence": "0",
    "issue": [],
    "driverId": "user_123",
    "warnings": [],
    "duration_ms": 4312.5
  }
}
```

### Failure response `422`
```json
{
  "success": false,
  "type": "driving_license",
  "total": 1,
  "succeeded": 0,
  "failed": 1,
  "index": 0,
  "filename": "license.jpg",
  "detection_confidence": null,
  "resultData": {
    "license_data": {
      "license_number": null,
      "full_name_english": null,
      "full_name_arabic": null,
      "date_of_birth": null,
      "date_of_issue": null,
      "date_of_expiry": null,
      "category": null,
      "nationality": null,
      "blood_group": null,
      "restrictions": null,
      "version": null,
      "issuing_country": null
    },
    "isValid": false,
    "sequence": "0",
    "issue": ["No license card detected."],
    "driverId": "user_123",
    "warnings": [],
    "duration_ms": 230.1
  }
}
```

---

## API — POST /verify-batch

Same form fields as `/verify` but `files` (plural) accepts multiple images at once.

| Field        | Type         | Required | Description                          |
|--------------|--------------|----------|--------------------------------------|
| `type`       | string       | yes      | Document type                        |
| `driverId`   | string       | yes      | Batch reference / user ID            |
| `files`      | image[]      | yes      | One or more license images           |
| `webhook_url`| string       | no       | Per-request callback URL             |

### curl
```bash
curl -X POST http://localhost:3000/verify-batch \
  -F "type=driving_license" \
  -F "driverId=batch_001" \
  -F "files=@license1.jpg" \
  -F "files=@license2.jpg"
```

### Response `200`
```json
{
  "success": true,
  "type": "driving_license",
  "total": 2,
  "succeeded": 2,
  "failed": 0,
  "results": [
    { "index": 0, "filename": "license1.jpg", "success": true, "detection_confidence": 0.97, "resultData": { ... } },
    { "index": 1, "filename": "license2.jpg", "success": true, "detection_confidence": 0.95, "resultData": { ... } }
  ]
}
```

---

## All endpoints

| Method | Path              | Description                          |
|--------|-------------------|--------------------------------------|
| GET    | `/`               | Liveness check                       |
| GET    | `/health`         | YOLO model + Together API status     |
| GET    | `/docs`           | Swagger UI                           |
| POST   | `/verify`         | Verify a single license image        |
| POST   | `/verify-batch`   | Verify multiple license images       |
| GET    | `/webhook/config` | View current webhook configuration   |
| POST   | `/webhook/config` | Update webhook URL/secret at runtime |
| POST   | `/webhook/test`   | Send a test payload to webhook URL   |

---

## Run pipeline from CLI (no API)

```bash
# Single image
python pipeline.py license.jpg

# Batch (all supported images in current directory)
python pipeline.py --batch

# Health check (Together API)
python pipeline.py --health
```





server command 

sudo apt install -y poppler-utils libgl1 libglib2.0-0
pip install -r requirements.txt
