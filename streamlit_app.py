"""
TrafficFlow AI - Traffic Optimization Demo
============================================
Two tabs:
  1. Bottleneck Detection (PS3) - Roboflow inference-sdk
  2. Traffic Monitoring (PS1) - YOLOv8 Colab results

Run: python -m streamlit run streamlit_app.py
"""

import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from inference_sdk import InferenceHTTPClient
import time
from collections import Counter

# ===== Page Config =====
st.set_page_config(
    page_title="TrafficFlow AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== Custom CSS =====
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    .stApp { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 2rem; color: white;
    }
    .main-header h1 { font-size: 2rem; font-weight: 800; margin: 0; color: white; }
    .main-header p { font-size: 1rem; opacity: 0.9; margin-top: 0.5rem; color: #e0e0ff; }
    .stat-card {
        background: linear-gradient(135deg, #1e2235 0%, #252a40 100%);
        border: 1px solid #2a2e45; border-radius: 12px; padding: 1.5rem; text-align: center;
    }
    .stat-card h3 { font-size: 2rem; font-weight: 800; margin: 0; color: #667eea; }
    .stat-card p { font-size: 0.85rem; color: #8b8fa3; margin-top: 0.25rem; }
    .detection-box { border-left: 4px solid; padding: 1rem 1.25rem; border-radius: 0 8px 8px 0; margin-bottom: 0.75rem; }
    .detection-box.critical { border-color: #f5576c; background: rgba(245, 87, 108, 0.08); }
    .detection-box.warning { border-color: #f5a623; background: rgba(245, 166, 35, 0.08); }
    .detection-box.info { border-color: #4facfe; background: rgba(79, 172, 254, 0.08); }
    .detection-box.success { border-color: #43e97b; background: rgba(67, 233, 123, 0.08); }
    div[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

# ===== Constants =====
API_URL = "https://serverless.roboflow.com"
API_KEY = st.secrets.get("ROBOFLOW_API_KEY", "fZJrRNq8wBaLvAXPRPkk")
MODEL_ID = "auto-generated-dataset-7-fbays/1"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(SCRIPT_DIR, "sample_images")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

CLASS_COLORS = {"people": (102,126,234), "parked vehicles": (245,87,108), "umbrella shops": (245,166,35), "potholes": (255,255,0)}
CLASS_SEVERITY = {"potholes": "critical", "parked vehicles": "warning", "umbrella shops": "warning", "people": "info"}
CLASS_ICONS = {"people": "🚶", "parked vehicles": "🚗", "umbrella shops": "☂️", "potholes": "🕳️"}


@st.cache_resource
def get_client():
    return InferenceHTTPClient(api_url=API_URL, api_key=API_KEY)


def run_inference(image_path, confidence=40):
    client = get_client()
    result = client.infer(image_path, model_id=MODEL_ID)
    predictions = result.get("predictions", [])
    filtered = [p for p in predictions if p.get("confidence", 0) * 100 >= confidence]
    return filtered, result


def annotate_image(img_rgb, predictions):
    img = img_rgb.copy()
    h, w = img.shape[:2]
    for pred in predictions:
        cls = pred.get("class", "unknown")
        conf = pred.get("confidence", 0)
        x, y = int(pred.get("x", 0)), int(pred.get("y", 0))
        bw, bh = int(pred.get("width", 0)), int(pred.get("height", 0))
        x1, y1 = max(0, x - bw//2), max(0, y - bh//2)
        x2, y2 = min(w, x + bw//2), min(h, y + bh//2)
        rgb = CLASS_COLORS.get(cls, (102, 126, 234))
        cv2.rectangle(img, (x1,y1), (x2,y2), rgb, 3)
        label = f"{cls} {conf:.0%}"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1-th-12), (x1+tw+8, y1), rgb, -1)
        cv2.putText(img, label, (x1+4, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    return img


def get_sample_images():
    if not os.path.exists(SAMPLE_DIR):
        return []
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    return sorted(f for f in os.listdir(SAMPLE_DIR) if os.path.splitext(f)[1].lower() in exts and '_detected' not in f)


def generate_alert_text(class_counts):
    alerts = []
    if class_counts.get("potholes", 0) > 0:
        alerts.append({"level":"critical","icon":"🚨","title":"POTHOLE HAZARD DETECTED",
            "desc":f"{class_counts['potholes']} pothole(s) found. Immediate maintenance required."})
    if class_counts.get("parked vehicles", 0) >= 3:
        alerts.append({"level":"critical","icon":"🅿️","title":"ILLEGAL PARKING CLUSTER",
            "desc":f"{class_counts['parked vehicles']} parked vehicle(s) causing lane obstruction. Alert sent to traffic police."})
    elif class_counts.get("parked vehicles", 0) > 0:
        alerts.append({"level":"warning","icon":"🚗","title":"Parked Vehicles Detected",
            "desc":f"{class_counts['parked vehicles']} parked vehicle(s) found. Monitoring for bottleneck."})
    if class_counts.get("umbrella shops", 0) > 0:
        alerts.append({"level":"warning","icon":"☂️","title":"ROAD ENCROACHMENT - Street Vendors",
            "desc":f"{class_counts['umbrella shops']} umbrella shop encroachment(s). Road width reduced."})
    if class_counts.get("people", 0) > 5:
        alerts.append({"level":"info","icon":"🚶","title":"High Pedestrian Activity",
            "desc":f"{class_counts['people']} pedestrian(s) detected. Speed advisory recommended."})
    return alerts


# ===== Sidebar =====
with st.sidebar:
    st.markdown("## 🚦 TrafficFlow AI")
    st.markdown("---")
    st.markdown("### ⚙️ PS3 Settings")
    confidence = st.slider("Confidence Threshold", 10, 95, 40, 5)
    show_json = st.checkbox("Show Raw JSON", value=False)
    show_alerts = st.checkbox("Generate Alerts", value=True)
    st.markdown("---")
    st.markdown(f"**PS3 Model:** `{MODEL_ID}`")
    st.markdown("**PS1 Model:** YOLOv8m (COCO)")


# ===== Top-Level Tabs =====
tab_bottleneck, tab_traffic = st.tabs(["🔍 Bottleneck Detection (PS3)", "🚗 Traffic Monitoring (PS1)"])


# ================================================================
# TAB 1 - BOTTLENECK DETECTION (PS3)
# ================================================================
with tab_bottleneck:
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Bottleneck Detection - Problem Statement 3</h1>
        <p>Real-time Roboflow AI detection • Illegal parking, road encroachments, potholes & pedestrian activity</p>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio("Mode", ["📷 Upload Image", "🖼️ Sample Gallery", "📁 Batch Detection"], horizontal=True)
    st.markdown("---")

    # ----- Upload Image -----
    if mode == "📷 Upload Image":
        st.markdown("### Upload a Traffic Image for Detection")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png","bmp","webp"])

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            temp_path = os.path.join(SCRIPT_DIR, "_temp_upload.jpg")
            cv2.imwrite(temp_path, img_bgr)

            with st.spinner("🔍 Running AI detection..."):
                t0 = time.time()
                predictions, raw_result = run_inference(temp_path, confidence=confidence)
                elapsed = time.time() - t0
            os.remove(temp_path)

            annotated = annotate_image(img_rgb, predictions)
            class_counts = Counter(p["class"] for p in predictions)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Detections", len(predictions))
            c2.metric("Classes", len(class_counts))
            c3.metric("Time", f"{elapsed:.2f}s")
            c4.metric("Confidence", f"≥{confidence}%")

            st.markdown("### Detection Results")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original**")
                st.image(img_rgb, use_container_width=True)
            with col2:
                st.markdown("**Detected**")
                st.image(annotated, use_container_width=True)

            if class_counts:
                st.markdown("### 📊 Breakdown")
                cols = st.columns(len(class_counts))
                for i, (cls, count) in enumerate(sorted(class_counts.items(), key=lambda x: -x[1])):
                    with cols[i]:
                        st.markdown(f'<div class="stat-card"><h3>{CLASS_ICONS.get(cls,"📦")} {count}</h3><p>{cls.title()}</p></div>', unsafe_allow_html=True)

            if show_alerts and class_counts:
                st.markdown("### 🚨 Alerts")
                for alert in generate_alert_text(class_counts):
                    st.markdown(f'<div class="detection-box {alert["level"]}"><strong>{alert["icon"]} {alert["title"]}</strong><br><span style="font-size:0.9rem">{alert["desc"]}</span></div>', unsafe_allow_html=True)

            if show_json:
                st.json(raw_result)

    # ----- Sample Gallery -----
    elif mode == "🖼️ Sample Gallery":
        st.markdown("### Sample Gallery - PS3 Dataset")
        samples = get_sample_images()
        if not samples:
            st.warning("No sample images found.")
        else:
            selected = st.selectbox("Select an image:", samples)
            with st.expander(f"📷 Thumbnails ({len(samples)} images)", expanded=False):
                for row in range(0, min(len(samples), 24), 6):
                    cols = st.columns(6)
                    for j, col in enumerate(cols):
                        idx = row + j
                        if idx < len(samples):
                            thumb = Image.open(os.path.join(SAMPLE_DIR, samples[idx]))
                            thumb.thumbnail((200, 200))
                            with col:
                                st.image(thumb, caption=samples[idx][:15], use_container_width=True)

            if selected and st.button("🔍 Run Detection", type="primary", use_container_width=True):
                image_path = os.path.join(SAMPLE_DIR, selected)
                img_bgr = cv2.imread(image_path)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                with st.spinner("🔍 Detecting..."):
                    t0 = time.time()
                    predictions, raw_result = run_inference(image_path, confidence=confidence)
                    elapsed = time.time() - t0

                annotated = annotate_image(img_rgb, predictions)
                class_counts = Counter(p["class"] for p in predictions)

                c1, c2, c3 = st.columns(3)
                c1.metric("Detections", len(predictions))
                c2.metric("Classes", len(class_counts))
                c3.metric("Time", f"{elapsed:.2f}s")

                col1, col2 = st.columns(2)
                with col1:
                    st.image(img_rgb, caption="Original", use_container_width=True)
                with col2:
                    st.image(annotated, caption="Detected", use_container_width=True)

                if class_counts:
                    cols = st.columns(len(class_counts))
                    for i, (cls, count) in enumerate(sorted(class_counts.items(), key=lambda x: -x[1])):
                        with cols[i]:
                            st.markdown(f'<div class="stat-card"><h3>{CLASS_ICONS.get(cls,"📦")} {count}</h3><p>{cls.title()}</p></div>', unsafe_allow_html=True)

                st.markdown("### 📋 Details")
                st.dataframe([{"Class": p["class"], "Confidence": f"{p['confidence']:.1%}", "X": int(p["x"]), "Y": int(p["y"])} for p in predictions], use_container_width=True)

                if show_alerts and class_counts:
                    for alert in generate_alert_text(class_counts):
                        st.markdown(f'<div class="detection-box {alert["level"]}"><strong>{alert["icon"]} {alert["title"]}</strong><br>{alert["desc"]}</div>', unsafe_allow_html=True)

                if show_json:
                    st.json(raw_result)

    # ----- Batch Detection -----
    elif mode == "📁 Batch Detection":
        st.markdown("### Batch Detection")
        samples = get_sample_images()
        if not samples:
            st.warning("No sample images found.")
        else:
            max_images = st.slider("Images to process", 1, min(len(samples), 50), 5)
            if st.button("🚀 Run Batch", type="primary", use_container_width=True):
                progress = st.progress(0)
                status = st.empty()
                all_results = []
                total_cc = Counter()
                total_time = 0

                for i, fname in enumerate(samples[:max_images]):
                    status.markdown(f"**Processing:** `{fname}` ({i+1}/{max_images})")
                    progress.progress((i+1) / max_images)
                    try:
                        t0 = time.time()
                        preds, _ = run_inference(os.path.join(SAMPLE_DIR, fname), confidence=confidence)
                        elapsed = time.time() - t0
                        total_time += elapsed
                        cc = Counter(p["class"] for p in preds)
                        total_cc.update(cc)
                        all_results.append({"image": fname, "detections": len(preds), "classes": dict(cc), "time": f"{elapsed:.2f}s"})
                    except Exception as e:
                        all_results.append({"image": fname, "error": str(e)})

                status.markdown("**✅ Done!**")
                total_dets = sum(r.get("detections", 0) for r in all_results if "error" not in r)
                ok = sum(1 for r in all_results if "error" not in r)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Processed", f"{ok}/{max_images}")
                c2.metric("Total Detections", total_dets)
                c3.metric("Total Time", f"{total_time:.1f}s")
                c4.metric("Avg/Image", f"{total_time/max(ok,1):.2f}s")

                if total_cc:
                    cols = st.columns(len(total_cc))
                    for i, (cls, count) in enumerate(sorted(total_cc.items(), key=lambda x: -x[1])):
                        with cols[i]:
                            st.markdown(f'<div class="stat-card"><h3>{CLASS_ICONS.get(cls,"📦")} {count}</h3><p>{cls.title()}</p></div>', unsafe_allow_html=True)

                    chart_data = []
                    for r in all_results:
                        if "error" not in r:
                            for cls in total_cc:
                                chart_data.append({"Image": r["image"][:12], "Class": cls, "Count": r["classes"].get(cls, 0)})
                    if chart_data:
                        st.bar_chart(pd.DataFrame(chart_data).pivot(index="Image", columns="Class", values="Count").fillna(0))

                st.dataframe(all_results, use_container_width=True)

                if show_alerts:
                    for alert in generate_alert_text(dict(total_cc)):
                        st.markdown(f'<div class="detection-box {alert["level"]}"><strong>{alert["icon"]} {alert["title"]}</strong><br>{alert["desc"]}</div>', unsafe_allow_html=True)


# ================================================================
# TAB 2 - TRAFFIC MONITORING (PS1 - YOLOv8 Results)
# ================================================================
with tab_traffic:
    st.markdown("""
    <div class="main-header">
        <h1>🚗 Traffic Monitoring - Problem Statement 1</h1>
        <p>YOLOv8 vehicle detection from drone footage • Lane-wise counting, congestion analysis & traffic flow</p>
    </div>
    """, unsafe_allow_html=True)

    csv_path = os.path.join(OUTPUT_DIR, "vehicle_counts_with_congestion.csv")
    csv_basic = os.path.join(OUTPUT_DIR, "vehicle_counts.csv")

    if os.path.exists(csv_path):
        df_ps1 = pd.read_csv(csv_path)
    elif os.path.exists(csv_basic):
        df_ps1 = pd.read_csv(csv_basic)
    else:
        df_ps1 = None

    if df_ps1 is not None:
        if 'frame_num' in df_ps1.columns:
            df_ps1 = df_ps1.sort_values('frame_num')

        # --- Stats ---
        st.markdown("### 📊 Overall Statistics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Frames", len(df_ps1))
        c2.metric("Total Vehicles", int(df_ps1['total_vehicles'].sum()))
        c3.metric("Avg/Frame", f"{df_ps1['total_vehicles'].mean():.1f}")
        c4.metric("Peak Count", int(df_ps1['total_vehicles'].max()))
        c5.metric("Total People", int(df_ps1['people'].sum()))

        # --- Vehicle Breakdown ---
        st.markdown("### 🚙 Vehicle Type Breakdown")
        vc1, vc2, vc3, vc4 = st.columns(4)
        with vc1:
            st.markdown(f'<div class="stat-card"><h3>🚗 {int(df_ps1["cars"].sum())}</h3><p>Cars</p></div>', unsafe_allow_html=True)
        with vc2:
            st.markdown(f'<div class="stat-card"><h3>🚌 {int(df_ps1["buses"].sum())}</h3><p>Buses</p></div>', unsafe_allow_html=True)
        with vc3:
            st.markdown(f'<div class="stat-card"><h3>🚛 {int(df_ps1["trucks"].sum())}</h3><p>Trucks</p></div>', unsafe_allow_html=True)
        with vc4:
            mc = int(df_ps1['motorcycles'].sum()) if 'motorcycles' in df_ps1.columns else 0
            st.markdown(f'<div class="stat-card"><h3>🏍️ {mc}</h3><p>Motorcycles</p></div>', unsafe_allow_html=True)

        # --- Congestion ---
        if 'congestion' in df_ps1.columns:
            st.markdown("### 🚦 Congestion Level Distribution")
            cong_counts = df_ps1['congestion'].value_counts()
            cong_cols = st.columns(4)
            cong_map = {'Low': ('🟢','success'), 'Medium': ('🟡','warning'), 'High': ('🔴','critical'), 'Critical': ('🔴','critical')}
            for i, level in enumerate(['Low','Medium','High','Critical']):
                count = cong_counts.get(level, 0)
                pct = count / len(df_ps1) * 100
                icon, css = cong_map.get(level, ('⚪','info'))
                with cong_cols[i]:
                    st.markdown(f'<div class="detection-box {css}"><strong>{icon} {level}</strong><br><span style="font-size:1.5rem;font-weight:800">{count}</span> frames ({pct:.1f}%)</div>', unsafe_allow_html=True)

        # --- Charts ---
        st.markdown("### 📈 Analysis Charts")
        for fname, title in [("sample_detections.png", "YOLOv8 Sample Detections"),
                              ("traffic_analysis.png", "Traffic Analysis"),
                              ("congestion_levels.png", "Congestion Levels Per Frame")]:
            fpath = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(fpath):
                st.markdown(f"#### {title}")
                st.image(Image.open(fpath), use_container_width=True)

        # --- Interactive Chart ---
        st.markdown("### 📉 Interactive Vehicle Count")
        if 'frame_num' in df_ps1.columns:
            st.line_chart(df_ps1.set_index('frame_num')[['cars','buses','trucks','people']])
        else:
            st.line_chart(df_ps1[['cars','buses','trucks','people']])

        # --- Data Table ---
        st.markdown("### 📋 Per-Frame Data")
        display_cols = [c for c in ['frame','total_vehicles','cars','motorcycles','buses','trucks','people','congestion'] if c in df_ps1.columns]
        st.dataframe(df_ps1[display_cols].reset_index(drop=True), use_container_width=True, height=400)

        st.download_button("📥 Download CSV", df_ps1.to_csv(index=False), "vehicle_counts.csv", "text/csv")

    else:
        st.warning("No PS1 output data found. Run the YOLOv8 Colab notebook first, then place output files in `demo-app/output/`.")
