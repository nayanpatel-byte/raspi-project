import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

CSV_FILE = "red_zone_log.csv"

# Create figure with 5 subplots
fig, axs = plt.subplots(3, 2, figsize=(10, 8))
axs = axs.flatten()

def update(frame):
    try:
        df = pd.read_csv(CSV_FILE)
        df.columns = df.columns.str.strip()

        if len(df) < 5:
            return

        # Time axis
        if "timestamp" in df.columns:
            time = df["timestamp"] - df["timestamp"].iloc[0]
        else:
            time = range(len(df))

        # Clear plots
        for ax in axs:
            ax.clear()

        # FPS
        if "fps" in df.columns:
            axs[0].plot(time, df["fps"])
            axs[0].set_title("FPS")

        # Processing Time
        proc_col = None
        for c in ["processing_time_ms", "proc_time", "processing_time"]:
            if c in df.columns:
                proc_col = c
                break
        if proc_col:
            axs[1].plot(time, df[proc_col])
            axs[1].set_title("Processing Time")

        # Polygon Area
        for c in ["polygon_area", "area"]:
            if c in df.columns:
                axs[2].plot(time, df[c])
                axs[2].set_title("Polygon Area")
                break

        # Variation
        for c in ["polygon_variation", "variation"]:
            if c in df.columns:
                axs[3].plot(time, df[c])
                axs[3].set_title("Variation")
                break

        # CPU
        for c in ["cpu_usage", "cpu"]:
            if c in df.columns:
                axs[4].plot(time, df[c])
                axs[4].set_title("CPU Usage")
                break

        for ax in axs:
            ax.grid()

    except Exception as e:
        print("Error:", e)

# Update every 1000 ms (1 sec)
ani = FuncAnimation(fig, update, interval=1000)

plt.tight_layout()
plt.show()