# -*- coding: utf-8 -*-
"""
Universal Spectrum Plotter - With Y-Axis Scaling & PGOPHER .dat Export
"""

import sys
import os
import csv
import json
import numpy as np
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# -----------------------------------------------------------------------------
# CUSTOM TOOLBAR
# -----------------------------------------------------------------------------
class CustomToolbar(NavigationToolbar2Tk):
    toolitems = [t for t in NavigationToolbar2Tk.toolitems if
                 t[0] in ('Home', 'Back', 'Forward', 'Pan', 'Zoom') or t[0] is None]

# -----------------------------------------------------------------------------
# ROBUST DATA READER
# -----------------------------------------------------------------------------
def parse_file_content(filename):
    x_list, y_list = [], []
    is_log_scale = False
    if not filename: return np.array([]), np.array([])
    try:
        with open(filename, 'r', errors='ignore') as f:
            lines = f.readlines()
            for line in lines[:20]:
                if "LogInt" in line or "Log Int" in line:
                    is_log_scale = True
                    break
            for line in lines:
                line = line.strip()
                if not line: continue 
                parts = line.replace(',', ' ').split()
                if len(parts) < 2: continue
                try:
                    x_list.append(float(parts[0]))
                    y_list.append(float(parts[1]))
                except ValueError:
                    continue
    except Exception as e:
        print(f"Error reading file: {e}")
        return np.array([]), np.array([])

    x_arr, y_arr = np.array(x_list), np.array(y_list)
    if is_log_scale and len(y_arr) > 0:
        try: y_arr = 10.0 ** y_arr
        except: pass
    return x_arr, y_arr

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
class SpectrumApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Universal Spectrum Plotter - Exp vs Pred")
        self.geometry("1150x850")

        self.top_x, self.top_y = [], []
        self.bot_x, self.bot_y = [], []
        
        self.selections = []      
        self.selected_freqs = []  
        self.peak_data = [] 

        # Keep track of Y-axis bounds for intensity zooming
        self.current_ymax = 1.1
        self.current_ymin = -1.1

        # --- 1. Top Controls (File & View) ---
        frame_files = tk.Frame(self, bg="#f0f0f0", bd=1, relief="raised")
        frame_files.pack(side=tk.TOP, fill=tk.X)

        tk.Button(frame_files, text="Load Exp (Top)", command=self.load_top, bg="white").pack(side=tk.LEFT, padx=10, pady=5)
        tk.Button(frame_files, text="Load Pred (Bottom)", command=self.load_bot, bg="white").pack(side=tk.LEFT, padx=10, pady=5)
        tk.Button(frame_files, text="Export Clean Image", command=self.export_clean_plot, bg="#d4edda").pack(side=tk.LEFT, padx=20, pady=5)
        tk.Button(frame_files, text="Clear Lines", command=self.clear_lines, bg="#ffcccc").pack(side=tk.RIGHT, padx=10, pady=5)

        self.lbl_info = tk.Label(frame_files, text="Load files to begin.", bg="#f0f0f0")
        self.lbl_info.pack(side=tk.LEFT, padx=10)

        # --- NEW: Intensity Scale Controls ---
        frame_scale = tk.Frame(self, bg="#e8f4f8", bd=1, relief="ridge")
        frame_scale.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        tk.Label(frame_scale, text="Exp Intensity (Top):", bg="#e8f4f8", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(10, 5))
        tk.Button(frame_scale, text="  -  ", command=self.dec_exp, bg="#ffcccc").pack(side=tk.LEFT, padx=2)
        tk.Button(frame_scale, text="  +  ", command=self.inc_exp, bg="#cce5ff").pack(side=tk.LEFT, padx=2)

        tk.Label(frame_scale, text="Pred Intensity (Bot):", bg="#e8f4f8", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(30, 5))
        tk.Button(frame_scale, text="  -  ", command=self.dec_pred, bg="#ffcccc").pack(side=tk.LEFT, padx=2)
        tk.Button(frame_scale, text="  +  ", command=self.inc_pred, bg="#cce5ff").pack(side=tk.LEFT, padx=2)

        tk.Button(frame_scale, text="Reset Y-Scale", command=self.reset_scales, bg="white").pack(side=tk.LEFT, padx=30)

        # --- 3. Advanced Peak Finder Controls ---
        frame_peaks = tk.Frame(self, bg="#ffeeba", bd=1, relief="ridge")
        frame_peaks.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        tk.Button(frame_peaks, text="Find All Bunny Ears (Zoom)", command=self.find_all_zoomed_pairs, bg="#ffc107", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=10, pady=5)

        tk.Label(frame_peaks, text="Sensitivity (%):", bg="#ffeeba").pack(side=tk.LEFT, padx=(10,0))
        self.prom_slider = tk.Scale(frame_peaks, from_=0.1, to=15.0, resolution=0.1, orient=tk.HORIZONTAL, bg="#ffeeba", length=120)
        self.prom_slider.set(2.0)
        self.prom_slider.pack(side=tk.LEFT, padx=5)

        tk.Label(frame_peaks, text="Max Split (MHz):", bg="#ffeeba").pack(side=tk.LEFT, padx=(10,0))
        self.max_split_var = tk.StringVar(value="1.0")
        tk.Entry(frame_peaks, textvariable=self.max_split_var, width=6).pack(side=tk.LEFT, padx=5)

        tk.Label(frame_peaks, text="Max Height Diff (%):", bg="#ffeeba").pack(side=tk.LEFT, padx=(10,0))
        self.max_hdiff_var = tk.StringVar(value="60")
        tk.Entry(frame_peaks, textvariable=self.max_hdiff_var, width=5).pack(side=tk.LEFT, padx=5)

        # --- 4. Plot Area ---
        frame_plot = tk.Frame(self)
        frame_plot.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Frequency (MHz)")
        self.ax.set_ylabel("Intensity")
        self.ax.grid(True, linestyle="--", alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_plot)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = CustomToolbar(self.canvas, frame_plot)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # --- 5. Bottom Log Box ---
        frame_bottom = tk.Frame(self, bg="#e1e1e1", height=150)
        frame_bottom.pack(side=tk.BOTTOM, fill=tk.X)
        
        tk.Label(frame_bottom, text="Measurement Log:", bg="#e1e1e1", font=("Arial", 10, "bold")).pack(side=tk.TOP, anchor="w", padx=10, pady=2)

        self.log_list = tk.Listbox(frame_bottom, height=6, font=("Courier", 10))
        scrollbar = tk.Scrollbar(frame_bottom, orient="vertical", command=self.log_list.yview)
        self.log_list.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        tk.Button(frame_bottom, text="Export Peaks", command=self.export_peaks_data, bg="#cce5ff", font=("Arial", 9, "bold")).pack(side=tk.RIGHT, padx=10, pady=10)
        tk.Button(frame_bottom, text="Copy Log", command=self.copy_to_clipboard, bg="#ddffdd").pack(side=tk.RIGHT, padx=5, pady=10)
        tk.Button(frame_bottom, text="Clear Log", command=self.clear_log, bg="#ffdddd").pack(side=tk.RIGHT, padx=5, pady=10)

    # ---------------------------------------------------------
    # INTENSITY SCALING LOGIC
    # ---------------------------------------------------------
    def inc_exp(self):
        self.current_ymax /= 1.5
        self.ax.set_ylim(self.current_ymin, self.current_ymax)
        self.canvas.draw()

    def dec_exp(self):
        self.current_ymax *= 1.5
        self.ax.set_ylim(self.current_ymin, self.current_ymax)
        self.canvas.draw()

    def inc_pred(self):
        self.current_ymin /= 1.5
        self.ax.set_ylim(self.current_ymin, self.current_ymax)
        self.canvas.draw()

    def dec_pred(self):
        self.current_ymin *= 1.5
        self.ax.set_ylim(self.current_ymin, self.current_ymax)
        self.canvas.draw()

    def reset_scales(self):
        self.current_ymax = 1.1
        self.current_ymin = -1.1 if len(self.bot_x) > 0 else -0.1
        self.ax.set_ylim(self.current_ymin, self.current_ymax)
        self.canvas.draw()

    # ---------------------------------------------------------
    # ROBUST MULTI-PEAK (BUNNY EAR) FINDER
    # ---------------------------------------------------------
    def find_all_zoomed_pairs(self):
        if len(self.top_x) == 0: return

        try:
            prominence_thresh = self.prom_slider.get() / 100.0
            max_split = float(self.max_split_var.get())
            max_hdiff = float(self.max_hdiff_var.get()) / 100.0
        except ValueError:
            print("Invalid peak parameters.")
            return

        x_min, x_max = self.ax.get_xlim()
        mask = (self.top_x >= x_min) & (self.top_x <= x_max)
        x_zoomed = self.top_x[mask]
        if len(x_zoomed) == 0: return

        y_norm_full = self.top_y / np.max(self.top_y)
        y_zoomed = y_norm_full[mask]

        peaks_indices, _ = find_peaks(y_zoomed, prominence=prominence_thresh)
        if len(peaks_indices) < 2: return

        found_freqs = x_zoomed[peaks_indices]
        found_y = y_zoomed[peaks_indices]

        for px, py in zip(found_freqs, found_y):
            mark, = self.ax.plot(px, py, 'r.', markersize=6, alpha=0.5)
            self.selections.append(mark)

        sorted_indices = sorted(range(len(found_freqs)), key=lambda idx: found_y[idx], reverse=True)
        unpaired = set(range(len(found_freqs)))
        pairs = []

        for i in sorted_indices:
            if i not in unpaired: continue
            
            best_j = -1
            best_dist = float('inf')
            
            for j in unpaired:
                if i == j: continue
                
                dist = abs(found_freqs[i] - found_freqs[j])
                if dist > max_split: continue
                
                h_diff = abs(found_y[i] - found_y[j]) / max(found_y[i], found_y[j])
                if h_diff > max_hdiff: continue
                
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
                    
            if best_j != -1:
                pairs.append({
                    'f1': min(found_freqs[i], found_freqs[best_j]),
                    'f2': max(found_freqs[i], found_freqs[best_j]),
                    'y1': found_y[i],
                    'y2': found_y[best_j]
                })
                unpaired.remove(i)
                unpaired.remove(best_j)

        for p in pairs:
            center_freq = (p['f1'] + p['f2']) / 2.0
            doppler_split = abs(p['f1'] - p['f2'])
            rel_intensity = (p['y1'] + p['y2']) / 2.0
            
            line = self.ax.axvline(x=center_freq, color='purple', linestyle='--', linewidth=1.5, alpha=0.8)
            self.selections.append(line)
            
            msg = f"C: {center_freq:.4f}\nInt: {rel_intensity:.3f}"
            t_lbl = self.ax.text(center_freq, rel_intensity + 0.05, msg, 
                                 color="purple", fontweight="bold", fontsize=9,
                                 ha="center", va="bottom",
                                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="purple"))
            self.selections.append(t_lbl)
            
            log_entry = f"AUTO Pair | Center: {center_freq:.4f} | Int: {rel_intensity:.4f} | Split: {doppler_split:.4f}"
            self.log_list.insert(tk.END, log_entry)
            self.log_list.yview(tk.END)

            self.peak_data.append({
                "Type": "Auto Bunny-Ear",
                "Center_Frequency_MHz": round(center_freq, 5),
                "Relative_Intensity": round(rel_intensity, 5),
                "Doppler_Split_MHz": round(doppler_split, 5)
            })

        self.canvas.draw()

    # ---------------------------------------------------------
    # MULTI-FORMAT EXPORT (INCLUDES NEW PGOPHER .DAT LOGIC)
    # ---------------------------------------------------------
    def export_peaks_data(self):
        if len(self.peak_data) == 0:
            print("No peaks to export!")
            return
            
        fname = filedialog.asksaveasfilename(
            defaultextension=".dat", 
            filetypes=[
                (" Stick Data", "*.dat"),  
                ("CSV File", "*.csv"), 
                ("Detailed Text", "*.txt"), 
                ("JSON File", "*.json")
            ]
        )
        if not fname: return

        try:
            ext = os.path.splitext(fname)[1].lower()
            
            # Sort peaks by frequency (lowest to highest) before exporting
            sorted_peaks = sorted(self.peak_data, key=lambda row: float(row['Center_Frequency_MHz']))

            if ext == '.csv':
                with open(fname, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=["Type", "Center_Frequency_MHz", "Relative_Intensity", "Doppler_Split_MHz"])
                    writer.writeheader()
                    for row in sorted_peaks: writer.writerow(row)
                    
            elif ext == '.txt':
                with open(fname, mode='w', encoding='utf-8') as file:
                    file.write(f"{'Type':<18} {'Center (MHz)':<15} {'Intensity':<12} {'Split (MHz)':<12}\n")
                    file.write("-" * 60 + "\n")
                    for row in sorted_peaks:
                        file.write(f"{row['Type']:<18} {row['Center_Frequency_MHz']:<15} {row['Relative_Intensity']:<12} {row['Doppler_Split_MHz']:<12}\n")
            
            # --- STICK EXPORT (.dat) - Formatted specifically for PGOPHER ---
            elif ext == '.dat':
                with open(fname, mode='w', encoding='utf-8') as file:
                    for row in sorted_peaks:
                        # Round the center frequency to exactly 3 decimal places
                        freq = round(float(row['Center_Frequency_MHz']), 3)
                        intensity = row['Relative_Intensity']
                        
                        if intensity == "N/A": 
                            intensity = 1.000
                        else:
                            intensity = float(intensity)
                        
                        # Write the zero anchor before, the peak, and the zero anchor after
                        file.write(f"{freq - 0.001:.3f} 0.000\n")
                        file.write(f"{freq:.3f} {intensity:.5f}\n")
                        file.write(f"{freq + 0.001:.3f} 0.000\n")
                        
            elif ext == '.json':
                with open(fname, mode='w', encoding='utf-8') as file:
                    json.dump(sorted_peaks, file, indent=4)
                    
            print(f"Successfully exported to {fname}")
        except Exception as e:
            print(f"Failed to export data: {e}")

    # ---------------------------------------------------------
    # EXPORT CLEAN IMAGE LOGIC
    # ---------------------------------------------------------
    def export_clean_plot(self):
        fname = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf")])
        if not fname: return
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
        
        if len(self.top_x) > 0:
            y_norm = self.top_y / np.max(self.top_y)
            ax.plot(self.top_x, y_norm, color="black", lw=1.0)
        if len(self.bot_x) > 0:
            max_y = np.max(self.bot_y) if np.max(self.bot_y) != 0 else 1
            y_norm = self.bot_y / max_y
            for f, i in zip(self.bot_x, y_norm):
                ax.vlines(f, 0, -i, color="red", linewidth=1.0)
        
        for artist in self.selections:
            if isinstance(artist, plt.Line2D):
                x = artist.get_xdata()[0]
                color, linestyle = artist.get_color(), artist.get_linestyle()
                ax.axvline(x=x, color=color, linestyle=linestyle, linewidth=1.5, alpha=0.8)
            elif isinstance(artist, plt.Text):
                x, y, text = artist.get_position()[0], artist.get_position()[1], artist.get_text()
                bbox = artist.get_bbox_patch()
                if bbox:
                    color = artist.get_color()
                    ax.text(x, y, text, color=color, fontweight="bold", ha="center", va="bottom",
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor=color))
                else:
                    ax.text(x, y, text, rotation=90, color="green", verticalalignment='bottom', backgroundcolor='white')

        ax.set_xlim(current_xlim)
        ax.set_ylim(current_ylim)
        ax.axhline(0, color='black', linewidth=1.0)
        ax.set_xlabel("Frequency (MHz)", fontsize=12)
        ax.set_ylabel("Intensity", fontsize=12)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)

    # ---------------------------------------------------------
    # LOADING & DRAWING
    # ---------------------------------------------------------
    def load_top(self):
        fname = filedialog.askopenfilename(title="Select Experimental File", filetypes=[("All Files", "*.*")])
        if fname:
            x, y = parse_file_content(fname)
            if len(x) > 0:
                self.top_x, self.top_y = x, y
                self.lbl_info.config(text=f"Loaded Top: {os.path.basename(fname)}")
                self.redraw_plot()

    def load_bot(self):
        fname = filedialog.askopenfilename(title="Select Prediction File", filetypes=[("All Files", "*.*")])
        if fname:
            x, y = parse_file_content(fname)
            if len(x) > 0:
                self.bot_x, self.bot_y = x, y
                self.lbl_info.config(text=f"Loaded Bottom: {os.path.basename(fname)}")
                self.redraw_plot()

    def redraw_plot(self):
        self.ax.clear()
        if len(self.top_x) > 0:
            y_norm = self.top_y / np.max(self.top_y)
            self.ax.plot(self.top_x, y_norm, color="black", lw=1.0, label="Experimental")
        if len(self.bot_x) > 0:
            max_y = np.max(self.bot_y) if np.max(self.bot_y) != 0 else 1
            y_norm = self.bot_y / max_y
            for f, i in zip(self.bot_x, y_norm):
                self.ax.vlines(f, 0, -i, color="red", linewidth=1.2)
            self.current_ymin = -1.1
        else:
            self.current_ymin = -0.1
            
        self.current_ymax = 1.1
        self.ax.set_ylim(self.current_ymin, self.current_ymax)

        self.ax.axhline(0, color='gray', linewidth=0.8)
        self.ax.grid(True, linestyle="--", alpha=0.3)
        self.ax.set_xlabel("Frequency (MHz)")
        self.canvas.draw()

    # ---------------------------------------------------------
    # MANUAL CLICK LOGIC
    # ---------------------------------------------------------
    def on_click(self, event):
        if event.inaxes != self.ax: return
        if event.button == 3: # Right Click
            freq = event.xdata
            line = self.ax.axvline(x=freq, color='green', linewidth=1.5, alpha=0.8)
            self.selections.append(line)
            t = self.ax.text(freq, 0.05, f"{freq:.4f}", rotation=90, color="green", 
                             verticalalignment='bottom', backgroundcolor='white')
            self.selections.append(t)
            self.selected_freqs.append(freq)
            
            if len(self.selected_freqs) % 2 == 0:
                f2, f1 = self.selected_freqs[-1], self.selected_freqs[-2]
                center_freq = (f1 + f2) / 2.0
                doppler_split = abs(f1 - f2)
                msg = f"Center: {center_freq:.4f}\nSplit: {doppler_split:.4f}"
                t_lbl = self.ax.text(center_freq, 0.5, msg, 
                                     color="blue", fontweight="bold", ha="center", va="bottom",
                                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="blue"))
                self.selections.append(t_lbl)
                
                log_entry = f"MANUAL Pair | Center: {center_freq:.4f} | Split: {doppler_split:.4f}  (F1: {f1:.4f}, F2: {f2:.4f})"
                self.log_list.insert(tk.END, log_entry)
                self.log_list.yview(tk.END)
                self.peak_data.append({
                    "Type": "Manual Pair", "Center_Frequency_MHz": round(center_freq, 5),
                    "Relative_Intensity": "N/A", "Doppler_Split_MHz": round(doppler_split, 5)
                })
            self.canvas.draw()

    def clear_lines(self):
        for artist in self.selections:
            try: artist.remove()
            except: pass
        self.selections = []
        self.selected_freqs = []
        self.canvas.draw()

    def clear_log(self):
        self.log_list.delete(0, tk.END)
        self.peak_data = [] 

    def copy_to_clipboard(self):
        items = self.log_list.get(0, tk.END)
        text_to_copy = "\n".join(items)
        self.clipboard_clear()
        self.clipboard_append(text_to_copy)
        self.update()

if __name__ == "__main__":
    app = SpectrumApp()
    app.mainloop()