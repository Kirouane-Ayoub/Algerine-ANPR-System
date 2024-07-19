import csv
import math
import time

import cv2
import cvzone
import numpy as np
import pandas as pd
import streamlit as st
from ocr import OCR
from tracker import Sort
from ultralytics import YOLO

# Initialize OCR object
ocro = OCR()

# Define the area of interest for license plate detection
area = np.array(
    [
        (1018, 382),
        (1019, 473),
        (1019, 476),
        (1016, 533),
        (1016, 532),
        (596, 514),
        (565, 514),
        (417, 514),
        (244, 503),
        (66, 506),
        (1, 509),
        (1, 483),
        (0, 370),
        (0, 345),
    ],
    np.int32,
)

# Initialize license plate list and counter
lplist = []
counter = 0


def get_device(select_device, st):
    if select_device == "GPU":
        return st.selectbox("Select GPU index:", (0, 1, 2))
    else:
        return "cpu"


def setup_output_video(cap, source):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    return cv2.VideoWriter(
        f'vid_results/{source.split(".")[0]}.mp4', fourcc, 10, (w, h)
    )


def save_crop(img, bbox, path):
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    cv2.imwrite(path, crop)


def write_log(text_result):
    with open("logs/log.csv", "a") as f:
        date_time = str(time.asctime())
        writer = csv.writer(f)
        writer.writerow([text_result, date_time])


def main():
    global counter
    with st.sidebar:
        st.image("icon.png", width=150)
        select_type_detect = st.selectbox("Detection from:", ("File", "Live"))
        save_lpcrops = st.selectbox(
            "Do you want to save license plate Crops?", ("Yes", "No")
        )
        save_cacrops = st.selectbox("Do you want to save Car Crops?", ("Yes", "No"))
        select_device = st.selectbox("Select compute Device:", ("CPU", "GPU"))
        save_output_video = st.radio("Save output video?", ("Yes", "No"))
        confd = st.slider(
            "Select threshold confidence value:",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
        )
        iou = st.slider(
            "Select Intersection over union (iou) value:",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
        )

    device_name = get_device(select_device, st)
    fps_reader = cvzone.FPS()
    class_names = ["license-plate", "vehicle"]

    tab1, tab2 = st.tabs(["Detection", "Log"])
    with tab1:
        if select_type_detect == "File":
            file = st.file_uploader("Select Your File:", type=["mp4", "mkv"])
            if file:
                source = file.name
                cap = cv2.VideoCapture(source)
        elif select_type_detect == "Live":
            source = st.text_input("Paste Your URL here and Click Enter")
            cap = cv2.VideoCapture(source)

        car_tracker = Sort(max_age=20)
        lp_tracker = Sort(max_age=20)
        model = YOLO("models/license-plates-us-eu-zgfga.pt")
        frame_window = st.image([])

        col_car, col_lp = st.columns(2)
        start, stop = st.columns(2)
        start_button = start.button("Click To Start")
        stop_button = stop.button("Click To Stop", key="ghedqLKHF")

        if start_button:
            with col_car:
                st.write("The Detection car:")
                car_frame = st.image([])
            with col_lp:
                st.write("The Detection License-plates:")
                lp_frame = st.image([])
                lptext = st.markdown("0")

            if save_output_video == "Yes":
                out = setup_output_video(cap, source)

            while True:
                try:
                    ret, img = cap.read()
                    if not ret:
                        break
                    img = cv2.resize(img, (1020, 600))
                    fps, img = fps_reader.update(
                        img, pos=(20, 50), color=(0, 255, 0), scale=2, thickness=3
                    )

                    if counter % 10 == 0:
                        results = model(img, conf=confd, iou=iou, device=device_name)
                        for result in results:
                            for box in result.boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                conf = math.ceil(box.conf[0] * 100)
                                clsi = int(box.cls[0])
                                w, h = x2 - x1, y2 - y1

                                cvzone.cornerRect(img, (x1, y1, w, h), l=7)
                                cvzone.putTextRect(
                                    img,
                                    f"{class_names[clsi]}",
                                    (max(0, x1), max(20, y1)),
                                    thickness=1,
                                    colorR=(0, 0, 255),
                                    scale=0.9,
                                    offset=3,
                                )

                                if clsi == 1:
                                    in_region = cv2.pointPolygonTest(
                                        area, (x2, y2), False
                                    )
                                    if in_region >= 0:
                                        detections_car = np.array(
                                            [[x1, y1, x2, y2, conf]]
                                        )
                                        result_track = car_tracker.update(
                                            detections_car
                                        )
                                        for (
                                            xca1,
                                            yca1,
                                            xca2,
                                            yca2,
                                            idca,
                                        ) in result_track:
                                            if save_cacrops == "Yes":
                                                save_crop(
                                                    img,
                                                    (xca1, yca1, xca2, yca2),
                                                    f"Cars_nump/{int(idca)}.jpg",
                                                )

                                if clsi == 0:
                                    try:
                                        lp_crop = img[y1:y2, x1:x2]
                                        detections_lp = np.array(
                                            [[x1, y1, x2, y2, conf]]
                                        )
                                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                                        in_area = cv2.pointPolygonTest(
                                            area, (cx, cy), False
                                        )
                                        if in_area >= 0:
                                            resultlp_track = lp_tracker.update(
                                                detections_lp
                                            )
                                            for (
                                                xlp1,
                                                ylp1,
                                                xlp2,
                                                ylp2,
                                                idlp,
                                            ) in resultlp_track:
                                                if save_lpcrops == "Yes":
                                                    save_crop(
                                                        img,
                                                        (xlp1, ylp1, xlp2, ylp2),
                                                        f"num_plate/{int(idlp)}.jpg",
                                                    )
                                                text_result = ocro.easyocr_fun(lp_crop)
                                                lptext.write(
                                                    f"The License-plates Number: {text_result}"
                                                )
                                                lplist.append(text_result)
                                                if lplist.count(text_result) == 4:
                                                    write_log(text_result)
                                                    lplist.clear()
                                                car_frame.image(
                                                    f"Cars_nump/{int(idca)}.jpg"
                                                )
                                                lp_frame.image(
                                                    f"num_plate/{int(idlp)}.jpg"
                                                )
                                    except:
                                        pass

                        frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        frame_window.image(frame)
                        if save_output_video == "Yes":
                            out.write(img)
                except:
                    pass
                counter += 1

            cap.release()
            if save_output_video == "Yes":
                out.release()
        if stop_button:
            try:
                cap.release()

            except:
                pass

    with tab2:
        dataf = pd.read_csv("logs/log.csv")
        st.dataframe(dataf)


if __name__ == "__main__":
    main()
