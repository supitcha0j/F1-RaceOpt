# รันด้วย gunicorn (production WSGI server) แทน Flask dev server ที่ debug=True
# ใช้ตอน python app.py เฉยๆ (ดู CLAUDE.md) — app:app คือ Flask instance ใน app.py
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

# $PORT มาจาก Render/Railway ตอน deploy จริง — ใช้ 5000 เป็นค่า default ตอนรันเองในเครื่อง
# ต้องใช้ sh -c เพื่อให้ ${PORT:-5000} ขยายค่าได้ (JSON exec-form ธรรมดาไม่รองรับ env var expansion)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 2 --timeout 120 app:app"]
