#!/usr/bin/env python3
import cv2
import numpy as np
import time
import csv
import os
import psutil

# ================= CONFIG =================
WIDTH, HEIGHT = 480, 360
FRAME_SKIP = 2

LOW_RED1 = np.array([0, 130, 100])
HIGH_RED1 = np.array([12, 255, 255])
LOW_RED2 = np.array([168, 130, 100])
HIGH_RED2 = np.array([180, 255, 255])

MIN_AREA = 1000
MAX_AREA = 200000
LOG_BATCH = 25

# ================= LOGGER =================
class CSVLogger:
    def __init__(self, path="red_zone_log.csv"):
        self.buffer = []
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            "timestamp","frame_id","fps","proc_ms",
            "red_pixels","red_ratio",
            "contours","valid_contours","zones",
            "zone_id","area","vertices","cx","cy",
            "cpu","mem"
        ])

    def log(self, row):
        self.buffer.append(row)
        if len(self.buffer) >= LOG_BATCH:
            self.writer.writerows(self.buffer)
            self.file.flush()
            self.buffer.clear()

    def close(self):
        if self.buffer:
            self.writer.writerows(self.buffer)
            self.file.flush()
        self.file.close()

# ================= MAIN =================
def main():

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("❌ Camera failed to open")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    logger = CSVLogger()
    process = psutil.Process(os.getpid())

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    frame_id = 0
    prev_time = time.time()

    print("✅ Detection Running (Single Window Mode)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % FRAME_SKIP != 0:
            frame_id += 1
            continue

        start = time.time()

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame_blur = cv2.blur(frame, (3,3))
        hsv = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2HSV)

        # ===== RED MASK =====
        mask = cv2.inRange(hsv, LOW_RED1, HIGH_RED1) + \
               cv2.inRange(hsv, LOW_RED2, HIGH_RED2)

        # Filters
        sat = hsv[:,:,1]
        val = hsv[:,:,2]

        mask = cv2.bitwise_and(mask, mask, mask=(sat > 120).astype(np.uint8)*255)
        mask = cv2.bitwise_and(mask, mask, mask=(val > 80).astype(np.uint8)*255)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        red_pixels = cv2.countNonZero(mask)
        red_ratio = red_pixels / (WIDTH * HEIGHT)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_AREA or area > MAX_AREA:
                continue

            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue

            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < 0.2:
                continue

            valid.append(c)

        zones = []
        for i, c in enumerate(valid[:5]):
            hull = cv2.convexHull(c)
            area = cv2.contourArea(hull)

            M = cv2.moments(hull)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
            else:
                x,y,w,h = cv2.boundingRect(hull)
                cx, cy = x+w//2, y+h//2

            zones.append((i, hull, area, len(hull), cx, cy))

        # ===== PERFORMANCE =====
        end = time.time()
        fps = 1/(end - prev_time) if (end-prev_time)>0 else 0
        prev_time = end
        proc_ms = (end-start)*1000

        cpu = process.cpu_percent()
        mem = process.memory_info().rss / (1024*1024)

        # ===== LOG =====
        for z in zones or [(-1, None, 0, 0, 0, 0)]:
            logger.log([
                time.time(), frame_id, round(fps,2), round(proc_ms,2),
                red_pixels, round(red_ratio,4),
                len(contours), len(valid), len(zones),
                z[0], round(z[2],2), z[3], z[4], z[5],
                cpu, round(mem,2)
            ])

        # ===== DISPLAY =====
        for z in zones:
            cv2.drawContours(frame, [z[1]], -1, (0,0,255), 2)
            cv2.circle(frame, (z[4], z[5]), 4, (0,255,0), -1)

        cv2.putText(frame, f"FPS:{fps:.1f}", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255),1)

        cv2.imshow("Red Zone Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        frame_id += 1

    logger.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()