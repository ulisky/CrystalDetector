import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from PIL import Image, ImageTk
import math
import statistics
import time
import os
import threading
import queue
import re # Needed for sorting frames correctly

# This import is from your original script. Make sure the .py file is in the same directory.
from train_resnet50_cosine import EmbeddingNet
#Hello Aiden


class CrystalTrackerApp:
    def __init__(self, master):
        self.master = master
        master.title("Crystal Detector and Tracker")
        master.geometry("1200x900")

        # --- CLASS VARIABLES ---
        self.video_path = None
        self.update_queue = queue.Queue()
        self.models_loaded = False
        self.yolo_model = None
        self.embedder_model = None
        self.stop_event = threading.Event()
        self.save_frames_var = tk.BooleanVar(value=True)
        self.draw_vectors_var = tk.BooleanVar(value=True)

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.device_info = "Apple Metal (MPS GPU)"
        else:
            self.device = torch.device("cpu")
            self.device_info = "CPU"
        print(f"Using device: {self.device_info}")

        # --- WIDGETS ---
        self.create_widgets()
        
        # This line will now work correctly
        self.master.after(100, self.check_queue)

    def create_widgets(self):
        top_frame = tk.Frame(self.master, padx=10, pady=10)
        top_frame.pack(fill=tk.X)

        self.select_button = tk.Button(top_frame, text="Select Video File", command=self.select_video_file)
        self.select_button.pack(side=tk.LEFT, padx=5)

        self.file_label = tk.Label(top_frame, text="No video selected", fg="gray", width=40, anchor="w")
        self.file_label.pack(side=tk.LEFT, padx=5)

        self.save_checkbox = ttk.Checkbutton(top_frame, text="Save Annotated Frames", variable=self.save_frames_var)
        self.save_checkbox.pack(side=tk.LEFT, padx=10)

        self.vectors_checkbox = ttk.Checkbutton(top_frame, text="Draw Velocity Vectors", variable=self.draw_vectors_var)
        self.vectors_checkbox.pack(side=tk.LEFT, padx=10)

        self.create_video_button = tk.Button(top_frame, text="Create Video", command=self.start_video_creation, state=tk.DISABLED)
        self.create_video_button.pack(side=tk.RIGHT, padx=5)

        self.stop_button = tk.Button(top_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.RIGHT, padx=5)

        self.start_button = tk.Button(top_frame, text="Start Analysis", command=self.start_processing, state=tk.DISABLED)
        self.start_button.pack(side=tk.RIGHT, padx=5)

        self.canvas = tk.Canvas(self.master, bg="black")
        self.canvas.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)
        bottom_frame = tk.Frame(self.master, padx=10, pady=10)
        bottom_frame.pack(fill=tk.X)
        self.progress_label = tk.Label(bottom_frame, text="Progress:")
        self.progress_label.pack(side=tk.LEFT)
        self.progress_bar = ttk.Progressbar(bottom_frame, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        self.progress_text_label = tk.Label(bottom_frame, text="", width=20)
        self.progress_text_label.pack(side=tk.LEFT, padx=5)
        stats_frame = tk.Frame(self.master, padx=10, pady=10)
        stats_frame.pack(side=tk.RIGHT, fill=tk.Y, before=bottom_frame)
        stats_label = tk.Label(stats_frame, text="Analysis Results:")
        stats_label.pack(anchor="w")
        self.stats_text = scrolledtext.ScrolledText(stats_frame, wrap=tk.WORD, width=40, height=15, state=tk.DISABLED)
        self.stats_text.pack(expand=True, fill=tk.BOTH)

    def start_processing(self):
        if not self.video_path: return
        self.stop_event.clear()
        self.start_button.config(state=tk.DISABLED)
        self.select_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.create_video_button.config(state=tk.DISABLED)
        self.progress_bar["value"] = 0
        self.progress_text_label.config(text="")
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, f"Using device: {self.device_info}\n\n")
        self.stats_text.insert(tk.END, "Processing... Please wait.\nModels are loading.")
        self.stats_text.config(state=tk.DISABLED)
        self.master.update_idletasks()
        self.processing_thread = threading.Thread(target=self.process_video_thread, args=(self.video_path,))
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def start_video_creation(self):
        self.create_video_button.config(state=tk.DISABLED)
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.insert(tk.END, "\n\nCreating video... Please wait.")
        self.stats_text.see(tk.END)
        self.stats_text.config(state=tk.DISABLED)
        self.master.update_idletasks()
        video_thread = threading.Thread(target=self.create_video_thread)
        video_thread.daemon = True
        video_thread.start()

    def stop_processing(self):
        self.stop_event.set()
        self.stop_button.config(state=tk.DISABLED)

    def select_video_file(self, *args, **kwargs):
        self.video_path = filedialog.askopenfilename(filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
        if self.video_path:
            self.file_label.config(text=os.path.basename(self.video_path), fg="black")
            self.start_button.config(state=tk.NORMAL)
            self.create_video_button.config(state=tk.DISABLED)
        else:
            self.file_label.config(text="No video selected", fg="gray")
            self.start_button.config(state=tk.DISABLED)

    # --- FIX HERE: The missing check_queue method is now restored ---
    def check_queue(self):
        try:
            message = self.update_queue.get_nowait()
            self.handle_message(message)
        except queue.Empty:
            pass
        finally:
            self.master.after(100, self.check_queue)
            
    def handle_message(self, message):
        msg_type = message.get("type")
        
        if msg_type == "progress":
            self.progress_bar["value"] = message.get("value")
            self.progress_text_label.config(text=message.get("text", ""))
            
        elif msg_type == "frame":
            frame = message.get("frame")
            canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if canvas_w > 1 and canvas_h > 1:
                h, w, _ = frame.shape
                scale = min(canvas_w/w, canvas_h/h)
                new_w, new_h = int(w*scale), int(h*scale)
                if new_w > 0 and new_h > 0:
                    resized_frame = cv2.resize(frame, (new_w, new_h))
                    img = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img)
                    self.photo = ImageTk.PhotoImage(image=img_pil)
                    self.canvas.create_image(canvas_w/2, canvas_h/2, image=self.photo, anchor=tk.CENTER)

        elif msg_type == "done":
            stats_report = message.get("report")
            self.stats_text.config(state=tk.NORMAL)
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, stats_report)
            self.stats_text.config(state=tk.DISABLED)
            
            self.start_button.config(state=tk.NORMAL)
            self.select_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

            if self.save_frames_var.get() and not self.stop_event.is_set():
                self.create_video_button.config(state=tk.NORMAL)
        
        elif msg_type == "video_done":
            report = message.get("report")
            self.stats_text.config(state=tk.NORMAL)
            self.stats_text.insert(tk.END, f"\n\n{report}")
            self.stats_text.see(tk.END)
            self.stats_text.config(state=tk.DISABLED)
            
    def process_video_thread(self, video_source):
        try:
            FRAME_SKIP = 2
            if not self.models_loaded:
                self.embedder_model = EmbeddingNet()
                checkpoint = torch.load("cosine_epoch30.pth", map_location=self.device)
                self.embedder_model.load_state_dict(checkpoint['state_dict'])
                self.embedder_model.to(self.device)
                self.embedder_model.eval()
                self.yolo_model = YOLO('YOLO_best.pt')
                self.yolo_model.to(self.device)
                self.models_loaded = True
            tracker = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.2, max_iou_distance=0.9)
            if self.save_frames_var.get():
                output_dir = "modified_images"
                os.makedirs(output_dir, exist_ok=True)
            cap = cv2.VideoCapture(video_source)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.progress_bar["maximum"] = total_frames
            curr_frame_idx, crys_ids = 0, {}
            while True:
                if self.stop_event.is_set(): break
                ret, frame = cap.read()
                if not ret: break
                img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                results = self.yolo_model(frame, conf=0.05, iou=0.2, verbose=False)[0]
                detections, pil_crops = [], []
                for index in range(len(results.obb.xywhr)):
                    x_min, y_min, x_max, y_max = results.obb.xyxy[index].cpu()
                    w, h = float(x_max - x_min), float(y_max - y_min)
                    detections.append(([x_min, y_min, w, h], float(results.obb.conf[index].cpu()), int(results.obb.cls[index].cpu())))
                    crop_box = tuple(map(int, (x_min, y_min, x_max, y_max)))
                    pil_crops.append(img_pil.crop(crop_box))
                if pil_crops:
                    extracted_features = self.embedder_model(pil_crops)
                    tracks = tracker.update_tracks(detections, embeds=extracted_features, frame=frame)
                    for track in [t for t in tracks if t.is_confirmed()]:
                        l, t, r, b = track.to_ltrb(orig=True)
                        w, h = r - l, b - t
                        crystal_id = track.track_id
                        
                        if crystal_id in crys_ids:
                            crys_ids[crystal_id]["positions"].append((l, t))
                            crys_ids[crystal_id]["size"].append(w * h)
                            x1, y1 = crys_ids[crystal_id]["positions"][-2]
                            velocity = math.sqrt((l - x1)**2 + (t - y1)**2)
                            crys_ids[crystal_id]["velocity"].append(velocity)
                            
                            if self.draw_vectors_var.get():
                                angle = math.atan2(t - y1, l - x1)
                                line_thickness = 2
                                length_scale_factor = 10
                                start_x = int(l + w / 2)
                                start_y = int(t + h / 2)
                                end_x = int(start_x + velocity * length_scale_factor * math.cos(angle))
                                end_y = int(start_y + velocity * length_scale_factor * math.sin(angle))
                                cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), line_thickness)
                        else:
                            crys_ids[crystal_id] = {"positions": [(l, t)], "velocity": [], "size": [w * h]}
                        
                        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (255, 0, 0), 2)
                        cv2.putText(frame, f"ID {crystal_id}", (int(l), int(t) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                if self.save_frames_var.get():
                    save_path = os.path.join(output_dir, f"frame_{curr_frame_idx:04d}.jpg")
                    cv2.imwrite(save_path, frame)
                
                progress_text = f"{curr_frame_idx + 1} / {total_frames}"
                self.update_queue.put({"type": "progress", "value": curr_frame_idx + 1, "text": progress_text})
                
                if curr_frame_idx % FRAME_SKIP == 0:
                    self.update_queue.put({"type": "frame", "frame": frame})

                curr_frame_idx += 1
            
            cap.release()

            final_report_text = ""
            if self.stop_event.is_set():
                final_report_text = f"--- Analysis Aborted by User ---\n\nFrames Processed: {curr_frame_idx}"
            else:
                velocity_list, unique_crystals_count = [], 0
                for cid in crys_ids:
                    if len(crys_ids[cid]["positions"]) > 3:
                        unique_crystals_count += 1
                        velocity_list.extend(crys_ids[cid]["velocity"])
                avg_vel = sum(velocity_list) / len(velocity_list) if velocity_list else 0
                std_dev = statistics.stdev(velocity_list) if len(velocity_list) > 1 else 0
                final_report_text = (f"--- Analysis Complete ---\n\n"
                                     f"Processing Device: {self.device_info}\n"
                                     f"Total Frames: {curr_frame_idx}\n"
                                     f"Unique Crystals Tracked: {unique_crystals_count}\n\n"
                                     f"--- Statistics ---\n"
                                     f"Average Velocity: {avg_vel:.2f} pixels/frame\n"
                                     f"Velocity Std Dev: {std_dev:.2f}\n")
            
            self.update_queue.put({"type": "done", "report": final_report_text})

        except Exception as e:
            error_report = f"An error occurred:\n\n{type(e).__name__}\n{e}"
            self.update_queue.put({"type": "done", "report": error_report})

    def create_video_thread(self):
        IMAGE_DIRECTORY = "modified_images"
        OUTPUT_VIDEO_NAME = f"{os.path.splitext(os.path.basename(self.video_path))[0]}_tracked.mp4"
        FRAMES_PER_SECOND = 24
        try:
            files = [f for f in os.listdir(IMAGE_DIRECTORY) if f.endswith('.jpg')]
            if not files:
                raise FileNotFoundError("No image files found in 'modified_images' directory.")
            def sort_key(f):
                return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', f)]
            files.sort(key=sort_key)
            first_image_path = os.path.join(IMAGE_DIRECTORY, files[0])
            frame = cv2.imread(first_image_path)
            height, width, _ = frame.shape
            frame_size = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            video_writer = cv2.VideoWriter(OUTPUT_VIDEO_NAME, fourcc, FRAMES_PER_SECOND, frame_size)
            for file in files:
                image_path = os.path.join(IMAGE_DIRECTORY, file)
                img = cv2.imread(image_path)
                if img is not None:
                    video_writer.write(img)
            video_writer.release()
            report = f"Video creation successful!\nSaved as: {OUTPUT_VIDEO_NAME}"
            self.update_queue.put({"type": "video_done", "report": report})
        except Exception as e:
            report = f"Video creation failed:\n{e}"
            self.update_queue.put({"type": "video_done", "report": report})

if __name__ == "__main__":
    root = tk.Tk()
    app = CrystalTrackerApp(root)
    root.mainloop()