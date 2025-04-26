# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
import cv2
import av
from typing import Any
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Cấu hình WebRTC TURN/STUN
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": "stun:stun.relay.metered.ca:80"},
        {"urls": "turn:global.relay.metered.ca:80", "username": "aee8f07d5e0f1cc64014c0d5", "credential": "LYdvYH8m1w3RISSO"},
        {"urls": "turn:global.relay.metered.ca:80?transport=tcp", "username": "aee8f07d5e0f1cc64014c0d5", "credential": "LYdvYH8m1w3RISSO"},
        {"urls": "turn:global.relay.metered.ca:443", "username": "aee8f07d5e0f1cc64014c0d5", "credential": "LYdvYH8m1w3RISSO"},
        {"urls": "turns:global.relay.metered.ca:443?transport=tcp", "username": "aee8f07d5e0f1cc64014c0d5", "credential": "LYdvYH8m1w3RISSO"},
    ]
})

# Bộ xử lý webcam
class YOLOProcessor:
    def __init__(self, model, conf, iou, selected_classes, enable_trk):
        self.model = model
        self.conf = conf
        self.iou = iou
        self.selected_classes = selected_classes
        self.enable_trk = enable_trk == "Yes"

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.enable_trk:
            results = self.model.track(img, conf=self.conf, iou=self.iou, classes=self.selected_classes, persist=True)
        else:
            results = self.model(img, conf=self.conf, iou=self.iou, classes=self.selected_classes)

        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# Giao diện và logic
class Inference:
    def __init__(self, **kwargs: Any):
        check_requirements("streamlit>=1.29.0")
        self.source = None
        self.enable_trk = False
        self.conf = 0.25
        self.iou = 0.45
        self.model = None
        self.selected_ind = []

        self.temp_dict = {"model": None, **kwargs}
        self.model_path = self.temp_dict.get("model")
        LOGGER.info(f"Ultralytics App Started: ✅ {self.temp_dict}")

    def web_ui(self):
        st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide")
        st.markdown("<style>MainMenu {visibility: hidden;}</style>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align:center; color:#FF64DA;'>Ultralytics YOLO Streamlit App 🚀</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align:center; color:#0066FF;'>Webcam and Video Object Detection</h4>", unsafe_allow_html=True)

    def sidebar(self):
        with st.sidebar:
            st.image("https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg", width=250)
            st.title("🔧 Settings")
            self.source = st.selectbox("Select Source", ("webcam", "video"))
            self.enable_trk = st.radio("Enable Tracking", ("Yes", "No"))
            self.conf = st.slider("Confidence", 0.0, 1.0, self.conf, 0.01)
            self.iou = st.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01)

    def configure_model(self):
        available_models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolo11")]
        if self.model_path:
            available_models.insert(0, self.model_path.split(".pt")[0])
        selected_model = st.sidebar.selectbox("Model", available_models)

        with st.spinner("Downloading and loading model..."):
            self.model = YOLO(f"{selected_model.lower()}.pt")
            class_names = list(self.model.names.values())
        st.success("✅ Model loaded successfully!")

        selected_classes = st.sidebar.multiselect("Classes to detect", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(cls) for cls in selected_classes]

    def run_webcam(self):
        st.subheader("📸 Webcam Stream")
        st.markdown("Scroll down to view the live feed from your webcam.")
        webrtc_streamer(
            key="yolo-webcam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=lambda: YOLOProcessor(
                model=self.model,
                conf=self.conf,
                iou=self.iou,
                selected_classes=self.selected_ind,
                enable_trk=self.enable_trk,
            ),
            async_processing=True,
        )

    def run_video(self):
        vid_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
        if vid_file is not None:
            with st.spinner("Reading video..."):
                g = io.BytesIO(vid_file.read())
                with open("uploaded_video.mp4", "wb") as out:
                    out.write(g.read())

            cap = cv2.VideoCapture("uploaded_video.mp4")
            if not cap.isOpened():
                st.error("❌ Could not open video file.")
                return

            st.subheader("📹 Uploaded Video Detection")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Frame**")
                org_frame = st.empty()
            with col2:
                st.markdown("**Detected Frame**")
                ann_frame = st.empty()

            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                results = self.model.track(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True) if self.enable_trk == "Yes" else self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                annotated = results[0].plot()

                org_frame.image(frame, channels="BGR")
                ann_frame.image(annotated, channels="BGR")

            cap.release()
            cv2.destroyAllWindows()

    def inference(self):
        self.web_ui()
        self.sidebar()
        self.configure_model()

        if self.source == "webcam":
            self.run_webcam()
        elif self.source == "video":
            self.run_video()


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else None
    Inference(model=model).inference()
