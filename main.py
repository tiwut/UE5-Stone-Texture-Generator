import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageOps
import threading
import time
import os

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class StoneGeneratorEngine:
    """
    Handles the mathematical generation of textures.
    Uses Vectorized NumPy operations for performance.
    """
    def __init__(self):
        pass

    def generate_noise_layer(self, shape, scale, seed, persistence, lacunarity, octaves):
        """
        Generates fractal noise using a vectorized approach.
        """
        width, height = shape
        rng = np.random.default_rng(seed)
        grid = np.zeros(shape, dtype=np.float32)
        
        freq = scale / min(width, height)
        amp = 1.0
        max_amp = 0.0

        for i in range(octaves):
            w_noise = int(width * freq) + 1
            h_noise = int(height * freq) + 1
            
            if w_noise > width: w_noise = width
            if h_noise > height: h_noise = height
            
            noise_base = rng.random((h_noise, w_noise), dtype=np.float32)
            
            img_temp = Image.fromarray(noise_base)
            img_temp = img_temp.resize((width, height), resample=Image.BILINEAR)
            layer = np.array(img_temp, dtype=np.float32)
            
            grid += layer * amp
            max_amp += amp
            
            amp *= persistence
            freq *= lacunarity
            
        return grid / max_amp

    def generate_maps(self, params, progress_callback):
        """
        Main generation logic pipeline.
        """
        w, h = params['width'], params['height']
        seed = params['seed']
        scale = params['scale']
        detail_octaves = params['detail_octaves']
        roughness_factor = params['roughness']
        
        contrast_val = params['contrast']
        depth_strength = params['depth_strength']
        
        progress_callback(10, "Generating Height Map (Base)...")
        
        base_noise = self.generate_noise_layer((w, h), scale, seed, 0.5, 2.0, detail_octaves)
        
        progress_callback(30, "Applying Domain Warping...")
        warp_x = self.generate_noise_layer((w, h), scale * 2, seed + 1, 0.5, 2.0, 2)
        warp_y = self.generate_noise_layer((w, h), scale * 2, seed + 2, 0.5, 2.0, 2)
        
        structure = (base_noise + 0.08 * warp_x + 0.08 * warp_y)
        
        structure = (structure - np.min(structure)) / (np.max(structure) - np.min(structure))

        progress_callback(40, "Refining Details...")
        structure = (structure - 0.5) * contrast_val + 0.5
        structure = np.clip(structure, 0.0, 1.0)

        progress_callback(50, "Generating Albedo...")
        color1 = np.array(params['color1'], dtype=np.float32)
        color2 = np.array(params['color2'], dtype=np.float32)
        
        s_expanded = structure[:, :, np.newaxis]
        
        albedo_arr = (s_expanded * color2 + (1.0 - s_expanded) * color1).astype(np.uint8)

        progress_callback(70, "Generating Normal Map (3D Calculation)...")
        
        gradient_y, gradient_x = np.gradient(structure)
        
        gradient_x *= depth_strength * 10.0 
        gradient_y *= depth_strength * 10.0
        
        normal_map = np.dstack((-gradient_x, gradient_y, np.ones_like(structure)))
        
        norm = np.linalg.norm(normal_map, axis=2)
        normal_map /= norm[:, :, np.newaxis]
        
        normal_map = ((normal_map + 1) * 0.5 * 255).astype(np.uint8)

        progress_callback(85, "Generating Roughness Map...")
        rough_arr = (1.0 - structure) * 255
        
        if roughness_factor != 1.0:
            rough_arr = (rough_arr - 127.5) * roughness_factor + 127.5
        
        rough_arr = np.clip(rough_arr, 0, 255).astype(np.uint8)

        progress_callback(95, "Finalizing Output...")
        
        height_arr = (structure * 255).astype(np.uint8)

        return {
            "albedo": Image.fromarray(albedo_arr, 'RGB'),
            "normal": Image.fromarray(normal_map, 'RGB'),
            "roughness": Image.fromarray(rough_arr, 'L'),
            "height": Image.fromarray(height_arr, 'L')
        }

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("UE5 Stone Texture Generator")
        self.geometry("1280x850")
        
        self.engine = StoneGeneratorEngine()
        self.preview_image = None

        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.main_area = ctk.CTkFrame(self, corner_radius=0)
        self.main_area.grid(row=0, column=1, sticky="nsew")
        
        self.preview_label = ctk.CTkLabel(self.main_area, text="Preview Area\n\nClick 'Generate Preview' to start.")
        self.preview_label.pack(expand=True, fill="both", padx=20, pady=20)

        self.progress_bar = ctk.CTkProgressBar(self.main_area)
        self.progress_bar.pack(fill="x", padx=20, pady=10)
        self.progress_bar.set(0)

        self.setup_controls()

    def setup_controls(self):
        pad_x = 15
        pad_y = 5
        
        title_font = ctk.CTkFont(size=18, weight="bold")
        
        ctk.CTkLabel(self.sidebar, text="Configuration", font=title_font).pack(pady=(20, 10))

        ctk.CTkLabel(self.sidebar, text="Resolution:").pack(padx=pad_x, anchor="w")
        self.res_var = ctk.StringVar(value="2048")
        self.res_combo = ctk.CTkComboBox(self.sidebar, values=["1024", "2048", "4096", "8192", "16384"], variable=self.res_var)
        self.res_combo.pack(padx=pad_x, pady=pad_y, fill="x")

        ctk.CTkLabel(self.sidebar, text="Noise Scale (Zoom):").pack(padx=pad_x, anchor="w")
        self.scale_slider = ctk.CTkSlider(self.sidebar, from_=2, to=30, number_of_steps=28)
        self.scale_slider.set(5)
        self.scale_slider.pack(padx=pad_x, pady=pad_y, fill="x")

        ctk.CTkLabel(self.sidebar, text="Detail Intensity (Contrast):").pack(padx=pad_x, anchor="w")
        self.contrast_slider = ctk.CTkSlider(self.sidebar, from_=0.5, to=3.0)
        self.contrast_slider.set(1.2)
        self.contrast_slider.pack(padx=pad_x, pady=pad_y, fill="x")

        ctk.CTkLabel(self.sidebar, text="3D Depth Strength:").pack(padx=pad_x, anchor="w")
        self.depth_slider = ctk.CTkSlider(self.sidebar, from_=1.0, to=20.0)
        self.depth_slider.set(5.0) 
        self.depth_slider.pack(padx=pad_x, pady=pad_y, fill="x")

        ctk.CTkLabel(self.sidebar, text="Micro-Details (Octaves):").pack(padx=pad_x, anchor="w")
        self.detail_slider = ctk.CTkSlider(self.sidebar, from_=1, to=8, number_of_steps=7)
        self.detail_slider.set(5)
        self.detail_slider.pack(padx=pad_x, pady=pad_y, fill="x")

        ctk.CTkLabel(self.sidebar, text="Colors (Hex):").pack(padx=pad_x, pady=(15,0), anchor="w")
        self.col_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.col_frame.pack(padx=pad_x, pady=pad_y, fill="x")
        
        self.col1_entry = ctk.CTkEntry(self.col_frame, width=100, placeholder_text="Color 1")
        self.col1_entry.insert(0, "#2b2b2b")
        self.col1_entry.pack(side="left", padx=(0, 5), expand=True, fill="x")
        
        self.col2_entry = ctk.CTkEntry(self.col_frame, width=100, placeholder_text="Color 2")
        self.col2_entry.insert(0, "#8a8a8a")
        self.col2_entry.pack(side="right", padx=(5, 0), expand=True, fill="x")

        ctk.CTkLabel(self.sidebar, text="Seed:").pack(padx=pad_x, anchor="w")
        self.seed_entry = ctk.CTkEntry(self.sidebar)
        self.seed_entry.insert(0, str(int(time.time())))
        self.seed_entry.pack(padx=pad_x, pady=pad_y, fill="x")

        self.btn_preview = ctk.CTkButton(self.sidebar, text="Generate Preview", command=lambda: self.start_generation(preview=True))
        self.btn_preview.pack(padx=pad_x, pady=(30, 10), fill="x")

        self.btn_export = ctk.CTkButton(self.sidebar, text="Export Textures", fg_color="#2CC985", text_color="black", hover_color="#25A56D", command=lambda: self.start_generation(preview=False))
        self.btn_export.pack(padx=pad_x, pady=0, fill="x")

        self.status_label = ctk.CTkLabel(self.sidebar, text="Ready", text_color="gray")
        self.status_label.pack(side="bottom", pady=10)

    def get_params(self, is_preview):
        try:
            res = int(self.res_var.get())
            if is_preview: res = 1024
            
            if res > 8192 and not is_preview:
                if not messagebox.askyesno("High Memory Warning", "16K Resolution requires >16GB RAM. It may take a few minutes. Continue?"):
                    return None

            return {
                'width': res,
                'height': res,
                'seed': int(self.seed_entry.get()),
                'scale': self.scale_slider.get(),
                'detail_octaves': int(self.detail_slider.get()),
                'roughness': 1.0,
                'contrast': self.contrast_slider.get(),
                'depth_strength': self.depth_slider.get(),
                'color1': self.hex_to_rgb(self.col1_entry.get()),
                'color2': self.hex_to_rgb(self.col2_entry.get())
            }
        except ValueError:
            messagebox.showerror("Input Error", "Please check your inputs (Seed must be numbers, Colors must be Hex).")
            return None

    def hex_to_rgb(self, h):
        try:
            h = h.lstrip('#')
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        except: return (128,128,128)

    def start_generation(self, preview):
        params = self.get_params(preview)
        if not params: return
        
        self.btn_preview.configure(state="disabled")
        self.btn_export.configure(state="disabled")
        
        threading.Thread(target=self.run_process, args=(params, preview), daemon=True).start()

    def run_process(self, params, preview):
        try:
            maps = self.engine.generate_maps(params, self.update_progress)
            
            if preview:
                self.show_preview(maps)
            else:
                self.save_maps(maps)
                
            self.update_progress(100, "Done")
        except Exception as e:
            print(e)
            self.update_progress(0, "Error")
            messagebox.showerror("Error", str(e))
        finally:
            self.btn_preview.configure(state="normal")
            self.btn_export.configure(state="normal")

    def update_progress(self, val, msg):
        self.progress_bar.set(val/100)
        self.status_label.configure(text=msg)

    def show_preview(self, maps):
        albedo = maps['albedo'].convert("RGBA")
        normal = maps['normal'].convert("RGBA")
        
        w_gui = self.main_area.winfo_width() - 50
        h_gui = self.main_area.winfo_height() - 50
        if w_gui < 100: w_gui = 500
        
        albedo.thumbnail((w_gui, h_gui))
        normal.thumbnail((w_gui, h_gui))
        
        final_preview = Image.blend(albedo, normal, 0.2)
        
        ctk_img = ctk.CTkImage(final_preview, size=final_preview.size)
        self.preview_label.configure(image=ctk_img, text="")

    def save_maps(self, maps):
        path = filedialog.askdirectory()
        if not path: return
        
        t = int(time.time())
        name = f"Stone_{t}"
        
        self.update_progress(90, "Writing files to disk...")
        maps['albedo'].save(f"{path}/{name}_Albedo.png")
        maps['normal'].save(f"{path}/{name}_Normal.png")
        maps['roughness'].save(f"{path}/{name}_Roughness.png")
        maps['height'].save(f"{path}/{name}_Height.png")
        
        messagebox.showinfo("Saved", "Textures generated successfully!")

if __name__ == "__main__":
    app = App()
    app.mainloop()