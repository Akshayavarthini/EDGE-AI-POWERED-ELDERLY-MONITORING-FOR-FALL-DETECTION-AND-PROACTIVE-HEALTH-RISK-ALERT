# EDGE-AI-POWERED-ELDERLY-MONITORING-FOR-FALL-DETECTION-AND-PROACTIVE-HEALTH-RISK-ALERT
Edge AI-based elderly monitoring system using TinyML on ESP32 for real-time fall detection and health risk prediction. Integrates sensors (SpO₂, temperature, respiration, gyroscope) for on-device analysis, ensuring low latency, privacy, and instant alerts to caregivers via IoT.

🧠 Edge AI-Powered Elderly Monitoring System

An intelligent Edge-AI based healthcare monitoring system designed for real-time fall detection and proactive health risk alerts using TinyML. The system runs on an ESP32 microcontroller and performs on-device analysis for faster response, improved privacy, and energy efficiency.

🚀 Features
📉 Real-time fall detection using motion sensors
❤️ Health monitoring (SpO₂, heart rate, temperature, respiration)
⚡ Edge AI (TinyML) for low latency & offline inference
🔔 Instant alerts to caregivers (WhatsApp/SMS)
🌐 IoT dashboard for remote monitoring
🔋 Energy-efficient and portable design
🛠️ Tech Stack
Hardware: ESP32, MAX30102, MPU6050, DHT11
Software: TinyML, TensorFlow Lite Micro
Domain: IoT, Embedded Systems, Healthcare AI
⚙️ System Architecture
Sensor Data Collection
Data Preprocessing & Feature Extraction
TinyML Model Inference (on ESP32)
Fall/Anomaly Detection
Alert Notification & IoT Dashboard Update
📊 Performance
✅ Accuracy: ~98%
📈 High precision and reliable detection of fall and abnormal conditions
📦 Project Structure
├── hardware/        # Circuit & sensor setup
├── model/           # Trained TinyML model
├── firmware/        # ESP32 code
├── data/            # Dataset & preprocessing
├── dashboard/       # IoT monitoring interface
└── README.md
🔧 Setup & Installation
Clone the repository
git clone https://github.com/Akshayavarthini/EDGE-AI-POWERED-ELDERLY-MONITORING-FOR-FALL-DETECTION-AND-PROACTIVE-HEALTH-RISK-ALERT.git
Upload code to ESP32 using Arduino IDE / PlatformIO
Connect sensors:
MAX30102 → SpO₂ & Heart Rate
MPU6050 → Fall Detection
DHT11 → Temperature & Humidity
Deploy TinyML model to ESP32
Run the system and monitor via IoT dashboard
📌 Applications
Elderly care & assisted living
Smart healthcare monitoring
Wearable health devices
🔮 Future Improvements
Integration of ECG sensors
Advanced deep learning models
Cloud analytics for long-term health prediction
Miniaturization for wearable devices
👨‍💻 Authors
Akshayavarthini B
Swathy Sree C S
Al Thameez S
