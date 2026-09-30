"""
main.py
-------
AI Data Preparation Tool
A four-tab desktop application for:
  1. Video → Images extraction
  2. Image Augmentation
  3. Auto Label  (Ultralytics YOLO)
  4. Dataset splitting (train / val / test)

Built with CustomTkinter for a modern look.
All heavy work runs in background threads to keep the UI responsive.
"""

import threading
import ctypes
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from core.augmenter import AugConfig, augment_folder
from core.auto_labeler import auto_label, generate_yaml, generate_classes_txt, get_model_classes # [UPDATED]
from core.dataset_splitter import split_dataset
from core.video_converter import extract_frames


# ---------------------------------------------------------------------------
# Global appearance
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT = "#3B8ED0"
ACCENT_HOVER = "#235B8E"
ACCENT_DARK = "#1A4971"
FONT_TITLE = ("Segoe UI Semibold", 15)
FONT_LABEL = ("Segoe UI", 12)
FONT_SMALL = ("Segoe UI", 11)
PAD = {"padx": 10, "pady": 5}


# ---------------------------------------------------------------------------
# Reusable UI helpers
# ---------------------------------------------------------------------------

def _make_folder_row(parent, label_text: str, row: int) -> ctk.StringVar:
    """Row with a label, entry, and Browse button. Returns the StringVar."""
    var = ctk.StringVar()
    ctk.CTkLabel(parent, text=label_text, font=FONT_LABEL).grid(
        row=row, column=0, sticky="w", **PAD
    )
    entry = ctk.CTkEntry(parent, textvariable=var, width=380, font=FONT_SMALL)
    entry.grid(row=row, column=1, sticky="ew", **PAD)
    ctk.CTkButton(
        parent, text="Browse", width=80,
        command=lambda: var.set(filedialog.askdirectory() or var.get()),
    ).grid(row=row, column=2, **PAD)
    return var


def _make_file_row(parent, label_text: str, row: int,
                   filetypes=(("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"),)) -> ctk.StringVar:
    """Row with a label, entry, and Browse-file button. Returns the StringVar."""
    var = ctk.StringVar()
    ctk.CTkLabel(parent, text=label_text, font=FONT_LABEL).grid(
        row=row, column=0, sticky="w", **PAD
    )
    entry = ctk.CTkEntry(parent, textvariable=var, width=380, font=FONT_SMALL)
    entry.grid(row=row, column=1, sticky="ew", **PAD)
    ctk.CTkButton(
        parent, text="Browse", width=80,
        command=lambda: var.set(
            filedialog.askopenfilename(filetypes=filetypes) or var.get()
        ),
    ).grid(row=row, column=2, **PAD)
    return var


def _make_section(parent, text: str) -> ctk.CTkLabel:
    lbl = ctk.CTkLabel(parent, text=f"── {text} ──", font=FONT_TITLE,
                        text_color=ACCENT)
    return lbl


def _make_augrow(parent, row: int, label: str, default: str,
                 enabled_var: tk.BooleanVar, entry_width: int = 120
                 ) -> tuple[ctk.CTkCheckBox, ctk.CTkEntry, ctk.StringVar]:
    """Checkbox + label + entry for an augmentation parameter row."""
    val_var = ctk.StringVar(value=default)

    cb = ctk.CTkCheckBox(parent, text="", variable=enabled_var, width=22)
    cb.grid(row=row, column=0, padx=(6, 2), pady=3, sticky="w")

    ctk.CTkLabel(parent, text=label, font=FONT_SMALL).grid(
        row=row, column=1, sticky="w", padx=(0, 8), pady=3
    )
    entry = ctk.CTkEntry(parent, textvariable=val_var,
                         width=entry_width, font=FONT_SMALL)
    entry.grid(row=row, column=2, sticky="w", pady=3)
    return cb, entry, val_var


# ---------------------------------------------------------------------------
# Tab 1 – Video to Images
# ---------------------------------------------------------------------------

class VideoTab(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.columnconfigure(1, weight=1)
        self._build()

    def _build(self):
        _make_section(self, "Video → Images").grid(
            row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(12, 4)
        )

        self.video_var = _make_file_row(self, "Video file:", row=1)
        self.output_var = _make_folder_row(self, "Output folder:", row=2)

        # Mode selection
        ctk.CTkLabel(self, text="Extract by:", font=FONT_LABEL).grid(
            row=3, column=0, sticky="w", **PAD
        )
        self.mode_var = ctk.StringVar(value="seconds")
        mode_frame = ctk.CTkFrame(self, fg_color="transparent")
        mode_frame.grid(row=3, column=1, sticky="w")
        ctk.CTkRadioButton(mode_frame, text="Every N seconds",
                           variable=self.mode_var, value="seconds",
                           font=FONT_SMALL).pack(side="left", padx=(0, 16))
        ctk.CTkRadioButton(mode_frame, text="Every N frames",
                           variable=self.mode_var, value="frames",
                           font=FONT_SMALL).pack(side="left")

        # Value entry
        ctk.CTkLabel(self, text="N value:", font=FONT_LABEL).grid(
            row=4, column=0, sticky="w", **PAD
        )
        self.value_var = ctk.StringVar(value="1")
        ctk.CTkEntry(self, textvariable=self.value_var,
                     width=120, font=FONT_SMALL).grid(
            row=4, column=1, sticky="w", **PAD
        )
        ctk.CTkLabel(self, text="seconds / frames", font=FONT_SMALL,
                     text_color="gray").grid(row=4, column=2, sticky="w")

        # Prefix
        ctk.CTkLabel(self, text="File prefix:", font=FONT_LABEL).grid(
            row=5, column=0, sticky="w", **PAD
        )
        self.prefix_var = ctk.StringVar(value="frame")
        ctk.CTkEntry(self, textvariable=self.prefix_var,
                     width=120, font=FONT_SMALL).grid(
            row=5, column=1, sticky="w", **PAD
        )

        # Progress bar
        self.progress = ctk.CTkProgressBar(self, width=460)
        self.progress.set(0)
        self.progress.grid(row=6, column=0, columnspan=3, padx=10,
                           pady=(14, 4), sticky="ew")

        self.status_lbl = ctk.CTkLabel(self, text="", font=FONT_SMALL,
                                        text_color="gray")
        self.status_lbl.grid(row=7, column=0, columnspan=3, **PAD)

        ctk.CTkButton(self, text="▶  Start Extraction", font=FONT_LABEL,
                      command=self._start).grid(
            row=8, column=0, columnspan=3, pady=16
        )

    def _start(self):
        video = self.video_var.get().strip()
        output = self.output_var.get().strip()
        mode = self.mode_var.get()
        try:
            value = float(self.value_var.get())
        except ValueError:
            messagebox.showerror("Error", "N value must be a number.")
            return

        if not video or not output:
            messagebox.showerror("Error", "Please select both a video file and an output folder.")
            return

        self.progress.set(0)
        self.status_lbl.configure(text="Running…", text_color="gray")

        def _run():
            try:
                def _cb(cur, total):
                    pct = cur / total if total else 0
                    self.progress.set(pct)
                    self.status_lbl.configure(
                        text=f"Frame {cur} / {total}"
                    )

                count = extract_frames(
                    video_path=video,
                    output_folder=output,
                    mode=mode,
                    value=value,
                    prefix=self.prefix_var.get().strip() or "frame",
                    progress_callback=_cb,
                )
                self.progress.set(1)
                self.status_lbl.configure(
                    text=f"✓  Done – {count} images saved.", text_color="#4CAF50"
                )
            except Exception as exc:
                self.status_lbl.configure(text=f"✗  {exc}", text_color="#F44336")

        threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Tab 2 – Image Augmentation
# ---------------------------------------------------------------------------

class AugTab(ctk.CTkScrollableFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.columnconfigure(2, weight=1)
        self._build()

    # ── helper: create a titled card frame ──────────────────────────────────
    def _make_card(self, title: str, enabled_var: tk.BooleanVar,
                   parent, row: int, col: int = 0, colspan: int = 1) -> ctk.CTkFrame:
        """
        Bordered card with a checkbox toggle in its header.
        Returns the inner content frame where parameters should be placed.
        """
        card = ctk.CTkFrame(parent, corner_radius=10,
                            border_width=1, border_color="#3a3a3a",
                            fg_color="#1e1e1e") # Deep container color
        card.grid(row=row, column=col, columnspan=colspan,
                  sticky="nsew", padx=8, pady=8)
        card.columnconfigure(0, weight=1)

        # Header row: checkbox + title
        hdr = ctk.CTkFrame(card, fg_color="#2a2a2a", corner_radius=8)
        hdr.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        hdr.columnconfigure(1, weight=1)

        ctk.CTkCheckBox(hdr, text="", variable=enabled_var, width=22,
                        hover_color=ACCENT_HOVER, fg_color=ACCENT).grid(
            row=0, column=0, padx=(10, 4), pady=8
        )
        ctk.CTkLabel(hdr, text=title, font=("Segoe UI Semibold", 13),
                     text_color=ACCENT).grid(
            row=0, column=1, sticky="w", pady=8
        )

        # Content area below header
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=12, pady=(6, 12))
        return body

    # ── helper: one param row inside a card ─────────────────────────────────
    @staticmethod
    def _card_row(body: ctk.CTkFrame, row: int, col_offset: int,
                  label: str, default: str) -> ctk.StringVar:
        """Label + entry in a 2-column card body. Returns the StringVar."""
        var = ctk.StringVar(value=default)
        body.columnconfigure(col_offset+1, weight=1) # Entry column stretches
        
        ctk.CTkLabel(body, text=label, font=FONT_SMALL,
                     text_color="#cccccc").grid(
            row=row, column=col_offset, sticky="w", padx=(0, 6), pady=4
        )
        ctk.CTkEntry(body, textvariable=var, height=30,
                     font=FONT_SMALL).grid(
            row=row, column=col_offset + 1, sticky="ew", pady=4, padx=(0, 10)
        )
        return var

    # ── main build ──────────────────────────────────────────────────────────
    def _build(self):
        # Configure self (the scrollable frame internal container)
        self.columnconfigure(0, weight=1)

        _make_section(self, "✨ Professional Augmentation Pipeline").grid(
            row=0, column=0, sticky="w", padx=12, pady=(15, 8)
        )

        # ── I/O Paths Section (Full width) ──────────────────────────────────
        io_card = ctk.CTkFrame(self, corner_radius=12, border_width=1, border_color="#333333")
        io_card.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 15))
        io_card.columnconfigure(0, weight=1)
        
        io_content = ctk.CTkFrame(io_card, fg_color="transparent")
        io_content.grid(row=0, column=0, sticky="ew", padx=15, pady=12)
        io_content.columnconfigure(1, weight=1)

        self.aug_input_var  = _make_folder_row(io_content, "Input Dataset:",  row=0)
        self.aug_output_var = _make_folder_row(io_content, "Output Result:", row=1)

        # ── Dual Column Parameter Grid ──────────────────────────────────────
        params_frame = ctk.CTkFrame(self, fg_color="transparent")
        params_frame.grid(row=2, column=0, sticky="nsew")
        params_frame.columnconfigure((0, 1), weight=1)

        # --- LEFT COLUMN: Geometry & Clarity ---
        left_col = ctk.CTkFrame(params_frame, fg_color="transparent")
        left_col.grid(row=0, column=0, sticky="nsew")
        left_col.columnconfigure(0, weight=1)

        # Rotation
        self.rot_enabled = tk.BooleanVar(value=False)
        rot_body = self._make_card("🔄  Spatial Rotation", self.rot_enabled, left_col, row=0)
        rot_body.columnconfigure((1, 3), weight=1)
        self.angle_vars = []
        for i in range(4):
            v = ctk.StringVar(value="0")
            r, char_col = divmod(i, 2)
            lbl = f"Angle {i+1}"
            ctk.CTkLabel(rot_body, text=lbl, font=FONT_SMALL, text_color="#aaaaaa").grid(row=r, column=char_col*2, sticky="w", padx=(0, 5))
            ctk.CTkEntry(rot_body, textvariable=v, width=60, height=28).grid(row=r, column=char_col*2 + 1, sticky="ew", padx=(0, 10), pady=4)
            self.angle_vars.append(v)

        # Blur
        self.motion_blur_enabled = tk.BooleanVar(value=False)
        self.gauss_blur_enabled  = tk.BooleanVar(value=False)
        blur_body = self._make_card("💧  Smart Blur", self.motion_blur_enabled, left_col, row=1)
        blur_body.columnconfigure((1, 3), weight=1)
        
        # Blur row: Motion
        ctk.CTkCheckBox(blur_body, text="Motion", variable=self.motion_blur_enabled, font=FONT_SMALL, width=20).grid(row=0, column=0, sticky="w")
        self.motion_blur_var = ctk.StringVar(value="7")
        ctk.CTkEntry(blur_body, textvariable=self.motion_blur_var, width=50, height=28).grid(row=0, column=1, sticky="ew", padx=(4, 15))
        
        # Blur row: Gaussian (reusing same card for clarity)
        ctk.CTkCheckBox(blur_body, text="Gauss", variable=self.gauss_blur_enabled, font=FONT_SMALL, width=20).grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.gauss_blur_var = ctk.StringVar(value="5")
        ctk.CTkEntry(blur_body, textvariable=self.gauss_blur_var, width=50, height=28).grid(row=1, column=1, sticky="ew", padx=(4, 15), pady=(10, 0))

        # --- RIGHT COLUMN: Color & Environment ---
        right_col = ctk.CTkFrame(params_frame, fg_color="transparent")
        right_col.grid(row=0, column=1, sticky="nsew")
        right_col.columnconfigure(0, weight=1)

        # Brightness & Contrast
        self.color_enabled = tk.BooleanVar(value=False)
        bc_body = self._make_card("☀  Lighting Controls", self.color_enabled, right_col, row=0)
        self.brightness_var = self._card_row(bc_body, 0, 0, "Brightness", "0.25")
        self.contrast_var   = self._card_row(bc_body, 1, 0, "Contrast",   "0.25")

        # HSV
        self.hsv_enabled = tk.BooleanVar(value=False)
        hsv_body = self._make_card("🎨  HSV Color Shift", self.hsv_enabled, right_col, row=1)
        self.hue_var = self._card_row(hsv_body, 0, 0, "Hue", "10")
        self.sat_var = self._card_row(hsv_body, 1, 0, "Sat", "20")
        self.val_var = self._card_row(hsv_body, 2, 0, "Val", "15")

        # Noise & Fog (horizontal layout inside card)
        self.gauss_noise_enabled = tk.BooleanVar(value=False)
        self.fog_enabled = tk.BooleanVar(value=False)
        env_card_body = self._make_card("🌫  Atmospheric Effects", self.gauss_noise_enabled, left_col, row=2)
        env_card_body.columnconfigure((0, 1), weight=1)
        
        # Sub-frames for Noise/Fog
        noise_f = ctk.CTkFrame(env_card_body, fg_color="#252525", corner_radius=6)
        noise_f.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        noise_f.columnconfigure(1, weight=1)
        ctk.CTkCheckBox(noise_f, text="Noise", variable=self.gauss_noise_enabled, font=FONT_SMALL, width=20).grid(row=0, column=0, padx=5, pady=5)
        self.noise_low_var  = ctk.StringVar(value="5")
        self.noise_high_var = ctk.StringVar(value="35")
        ctk.CTkEntry(noise_f, textvariable=self.noise_low_var, width=40, height=24).grid(row=0, column=1, sticky="ew", padx=2)
        ctk.CTkEntry(noise_f, textvariable=self.noise_high_var, width=40, height=24).grid(row=0, column=2, sticky="ew", padx=5)

        fog_f = ctk.CTkFrame(env_card_body, fg_color="#252525", corner_radius=6)
        fog_f.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        fog_f.columnconfigure(1, weight=1)
        ctk.CTkCheckBox(fog_f, text="Fog", variable=self.fog_enabled, font=FONT_SMALL, width=20).grid(row=0, column=0, padx=5, pady=5)
        self.fog_low_var  = ctk.StringVar(value="0.05")
        self.fog_high_var = ctk.StringVar(value="0.20")
        ctk.CTkEntry(fog_f, textvariable=self.fog_low_var, width=45, height=24).grid(row=0, column=1, sticky="ew", padx=2)
        ctk.CTkEntry(fog_f, textvariable=self.fog_high_var, width=45, height=24).grid(row=0, column=2, sticky="ew", padx=5)

        # ── Bottom Action Area ──────────────────────────────────────────────
        self.aug_progress = ctk.CTkProgressBar(self, height=12)
        self.aug_progress.set(0)
        self.aug_progress.grid(row=3, column=0, padx=15, pady=(25, 5), sticky="ew")

        self.aug_status = ctk.CTkLabel(self, text="", font=FONT_SMALL, text_color="#888888")
        self.aug_status.grid(row=4, column=0, pady=(0, 10))

        ctk.CTkButton(self, text="🚀 Launch Augmentation Pipeline", font=("Segoe UI Bold", 13),
                      height=45, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                      command=self._start).grid(row=5, column=0, pady=(0, 30), sticky="n")



    def _build_config(self) -> AugConfig:
        """Read GUI values and build an AugConfig instance."""
        def _f(var): return float(var.get())
        def _i(var): return int(var.get())

        angles = []
        for v in self.angle_vars:
            try:
                angles.append(float(v.get()))
            except ValueError:
                angles.append(0.0)

        return AugConfig(
            rotation_enabled=self.rot_enabled.get(),
            rotation_angles=angles,
            color_enabled=self.color_enabled.get(),
            brightness_limit=_f(self.brightness_var),
            contrast_limit=_f(self.contrast_var),
            hsv_enabled=self.hsv_enabled.get(),
            hue_shift_limit=_i(self.hue_var),
            sat_shift_limit=_i(self.sat_var),
            val_shift_limit=_i(self.val_var),
            motion_blur_enabled=self.motion_blur_enabled.get(),
            motion_blur_limit=_i(self.motion_blur_var),
            gaussian_blur_enabled=self.gauss_blur_enabled.get(),
            gaussian_blur_limit=_i(self.gauss_blur_var),
            gauss_noise_enabled=self.gauss_noise_enabled.get(),
            gauss_noise_var_limit=(_i(self.noise_low_var), _i(self.noise_high_var)),
            fog_enabled=self.fog_enabled.get(),
            fog_coef_lower=_f(self.fog_low_var),
            fog_coef_upper=_f(self.fog_high_var),
        )

    def _start(self):
        inp = self.aug_input_var.get().strip()
        out = self.aug_output_var.get().strip()
        if not inp or not out:
            messagebox.showerror("Error", "Please select both input and output folders.")
            return

        try:
            cfg = self._build_config()
        except ValueError as exc:
            messagebox.showerror("Config error", str(exc))
            return

        self.aug_progress.set(0)
        self.aug_status.configure(text="Running…", text_color="gray")

        def _run():
            try:
                def _cb(cur, total):
                    self.aug_progress.set(cur / total if total else 0)
                    self.aug_status.configure(text=f"Image {cur} / {total}")

                count = augment_folder(inp, out, cfg, progress_callback=_cb)
                self.aug_progress.set(1)
                self.aug_status.configure(
                    text=f"✓  Done – {count} images written.", text_color="#4CAF50"
                )
            except Exception as exc:
                self.aug_status.configure(text=f"✗  {exc}", text_color="#F44336")

        threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Tab 3 – Dataset Splitter
# ---------------------------------------------------------------------------

class SplitTab(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.columnconfigure(1, weight=1)
        self._build()

    def _build(self):
        _make_section(self, "Train / Val / Test Split").grid(
            row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(12, 4)
        )
        self.src_var = _make_folder_row(
            self, "Source folder\n(contains images/ & labels/):", row=1
        )
        self.out_var = _make_folder_row(self, "Output folder:", row=2)

        # ── Train ratio slider ──
        ctk.CTkLabel(self, text="Train ratio:", font=FONT_LABEL).grid(
            row=3, column=0, sticky="w", **PAD
        )
        self.train_ratio_var = ctk.DoubleVar(value=0.7)
        train_slider_frame = ctk.CTkFrame(self, fg_color="transparent")
        train_slider_frame.grid(row=3, column=1, columnspan=2, sticky="w")
        ctk.CTkSlider(
            train_slider_frame, from_=0.5, to=0.9, number_of_steps=40,
            variable=self.train_ratio_var,
            command=self._update_split_label,
        ).pack(side="left", padx=(0, 10))

        # ── Val ratio slider ──
        ctk.CTkLabel(self, text="Val ratio:", font=FONT_LABEL).grid(
            row=4, column=0, sticky="w", **PAD
        )
        self.val_ratio_var = ctk.DoubleVar(value=0.2)
        val_slider_frame = ctk.CTkFrame(self, fg_color="transparent")
        val_slider_frame.grid(row=4, column=1, columnspan=2, sticky="w")
        ctk.CTkSlider(
            val_slider_frame, from_=0.05, to=0.4, number_of_steps=35,
            variable=self.val_ratio_var,
            command=self._update_split_label,
        ).pack(side="left", padx=(0, 10))

        # Live split summary label
        self.split_lbl = ctk.CTkLabel(
            self, text=self._format_split_label(), font=FONT_SMALL,
            text_color=ACCENT,
        )
        self.split_lbl.grid(row=5, column=0, columnspan=3, padx=10, pady=(0, 4))

        # ── Seed ──
        ctk.CTkLabel(self, text="Random seed:", font=FONT_LABEL).grid(
            row=6, column=0, sticky="w", **PAD
        )
        self.seed_var = ctk.StringVar(value="42")
        ctk.CTkEntry(self, textvariable=self.seed_var, width=90,
                     font=FONT_SMALL).grid(row=6, column=1, sticky="w", **PAD)

        # ── Progress ──
        self.split_progress = ctk.CTkProgressBar(self, width=460)
        self.split_progress.set(0)
        self.split_progress.grid(row=7, column=0, columnspan=3, padx=10,
                                  pady=(14, 4), sticky="ew")
        self.split_status = ctk.CTkLabel(self, text="", font=FONT_SMALL,
                                          text_color="gray")
        self.split_status.grid(row=8, column=0, columnspan=3, **PAD)

        ctk.CTkButton(self, text="▶  Shuffle & Split", font=FONT_LABEL,
                      command=self._start).grid(
            row=9, column=0, columnspan=3, pady=16
        )

    def _format_split_label(self) -> str:
        """Compute the three-way percentage string, clipping test to ≥ 0."""
        train = round(self.train_ratio_var.get(), 2)
        val   = round(self.val_ratio_var.get(), 2)
        test  = max(0.0, round(1.0 - train - val, 2))
        return (f"  Train {int(train * 100)}%  │  "
                f"Val {int(val * 100)}%  │  "
                f"Test {int(test * 100)}%")

    def _update_split_label(self, val=None):
        self.split_lbl.configure(text=self._format_split_label())

    def _start(self):
        src = self.src_var.get().strip()
        out = self.out_var.get().strip()
        if not src or not out:
            messagebox.showerror("Error", "Please select both source and output folders.")
            return
        try:
            seed = int(self.seed_var.get())
        except ValueError:
            messagebox.showerror("Error", "Seed must be an integer.")
            return

        train_ratio = round(self.train_ratio_var.get(), 3)
        val_ratio   = round(self.val_ratio_var.get(), 3)

        if train_ratio + val_ratio >= 1.0:
            messagebox.showerror(
                "Invalid ratios",
                "Train + Val ratio must be less than 1.0 so that a test set exists."
            )
            return

        self.split_progress.set(0)
        self.split_status.configure(text="Running…", text_color="gray")

        def _run():
            try:
                result = split_dataset(
                    source_folder=src,
                    output_folder=out,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    seed=seed,
                    progress_callback=lambda c, t: (
                        self.split_progress.set(c / t if t else 0),
                        self.split_status.configure(text=f"Copying {c} / {t}"),
                    ),
                )
                self.split_progress.set(1)
                self.split_status.configure(
                    text=(f"✓  Done – "
                          f"train: {result['train']},  "
                          f"val: {result['val']},  "
                          f"test: {result['test']} pairs."),
                    text_color="#4CAF50",
                )
            except Exception as exc:
                self.split_status.configure(text=f"✗  {exc}", text_color="#F44336")

        threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# Tab 3 – Auto Label
# ---------------------------------------------------------------------------

MODEL_PRESETS = [
    # YOLOv8
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    # YOLOv11 (Official names have no 'v')
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    # YOLOv12 (Official names have no 'v')
    "yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt",
    # YOLOv26 (Experimental/Preview names)
    "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt",
]


class AutoLabelTab(ctk.CTkScrollableFrame):
    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self.columnconfigure(1, weight=1)
        self.class_checkboxes = {}  # {id: CTkCheckBox}
        self._build()

    def _build(self):
        _make_section(self, "Auto Label  (Ultralytics YOLO)").grid(
            row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(12, 4)
        )

        # I/O paths
        io_frame = ctk.CTkFrame(self, corner_radius=8,
                                border_width=1, border_color="#3a3a3a")
        io_frame.columnconfigure(1, weight=1)
        io_frame.grid(row=1, column=0, columnspan=3, sticky="ew",
                      padx=8, pady=(0, 8))
        self.al_input_var  = _make_folder_row(io_frame, "Images folder:",  row=0)
        self.al_output_var = _make_folder_row(io_frame, "Labels output:",  row=1)

        # Model selection card
        model_card = ctk.CTkFrame(self, corner_radius=8,
                                  border_width=1, border_color="#3a3a3a")
        model_card.columnconfigure(1, weight=1)
        model_card.grid(row=2, column=0, columnspan=3, sticky="ew",
                        padx=8, pady=(0, 8))

        model_hdr = ctk.CTkFrame(model_card, fg_color="#2a2a2a", corner_radius=6)
        model_hdr.grid(row=0, column=0, columnspan=3, sticky="ew")
        ctk.CTkLabel(model_hdr, text="🤖  Model", font=("Segoe UI Semibold", 12),
                     text_color=ACCENT).grid(row=0, column=0, sticky="w",
                                             padx=12, pady=6)

        model_body = ctk.CTkFrame(model_card, fg_color="transparent")
        model_body.columnconfigure(1, weight=1)
        model_body.grid(row=1, column=0, columnspan=3, sticky="ew",
                        padx=12, pady=(6, 10))

        # Model Type Toggle
        self.model_type_var = ctk.StringVar(value="Preset")
        self.type_toggle = ctk.CTkSegmentedButton(
            model_body, values=["Preset", "Custom"],
            variable=self.model_type_var,
            command=self._toggle_model_type,
            font=FONT_SMALL, height=28
        )
        self.type_toggle.grid(row=0, column=0, columnspan=3, pady=(0, 12), sticky="w")

        # Preset row (row 1)
        self.preset_row = ctk.CTkFrame(model_body, fg_color="transparent")
        self.preset_row.columnconfigure(1, weight=1)
        self.preset_row.grid(row=1, column=0, columnspan=3, sticky="ew")
        
        ctk.CTkLabel(self.preset_row, text="Select model:",
                     font=FONT_SMALL, text_color="#bbbbbb").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.model_preset_var = ctk.StringVar(value=MODEL_PRESETS[0])
        ctk.CTkOptionMenu(self.preset_row, values=MODEL_PRESETS,
                          variable=self.model_preset_var,
                          font=FONT_SMALL, width=160).grid(
            row=0, column=1, sticky="w"
        )
        ctk.CTkLabel(self.preset_row, text="(downloads if needed)",
                     font=FONT_SMALL, text_color="gray").grid(
            row=0, column=2, sticky="w", padx=(10, 0)
        )

        # Custom row (row 2)
        self.custom_row = ctk.CTkFrame(model_body, fg_color="transparent")
        self.custom_row.columnconfigure(0, weight=1)
        
        self.custom_model_var = ctk.StringVar()
        ctk.CTkEntry(self.custom_row, textvariable=self.custom_model_var,
                     font=FONT_SMALL, placeholder_text="Path to .pt file...").grid(
            row=0, column=0, sticky="ew", padx=(0, 8)
        )
        ctk.CTkButton(
            self.custom_row, text="Browse", width=80,
            command=lambda: self.custom_model_var.set(
                filedialog.askopenfilename(filetypes=[("YOLO weights", "*.pt")]) 
                or self.custom_model_var.get()
            ),
        ).grid(row=0, column=1)

        # Action Buttons row (row 3)
        self.load_btn = ctk.CTkButton(
            model_body, text="📋 Load Classes from Model", font=FONT_SMALL,
            fg_color="transparent", border_width=1,
            hover_color="#333333",
            command=self._fetch_classes
        )
        self.load_btn.grid(row=3, column=0, columnspan=3, pady=(12, 0), sticky="w")

        # Checklist (row 4)
        self.class_list_frame = ctk.CTkScrollableFrame(model_body, height=220,
                                                       fg_color="#181818", border_width=1,
                                                       border_color="#333333")
        self.class_list_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        self.class_list_frame.columnconfigure((0, 1, 2), weight=1)
        
        ctk.CTkLabel(self.class_list_frame, text="(Select model then 'Load Classes')",
                     font=FONT_SMALL, text_color="gray").grid(row=0, column=0, columnspan=3, pady=10)
        
        # Initial call to set visibility
        self.after(100, lambda: self._toggle_model_type("Preset"))

        # Threshold card
        thresh_card = ctk.CTkFrame(self, corner_radius=8,
                                   border_width=1, border_color="#3a3a3a")
        thresh_card.columnconfigure((1, 3), weight=1)
        thresh_card.grid(row=3, column=0, columnspan=3, sticky="ew",
                         padx=8, pady=(0, 8))

        thresh_hdr = ctk.CTkFrame(thresh_card, fg_color="#2a2a2a", corner_radius=6)
        thresh_hdr.grid(row=0, column=0, columnspan=5, sticky="ew")
        ctk.CTkLabel(thresh_hdr, text="⚙  Thresholds",
                     font=("Segoe UI Semibold", 12),
                     text_color=ACCENT).grid(row=0, column=0, sticky="w",
                                             padx=12, pady=6)

        thresh_body = ctk.CTkFrame(thresh_card, fg_color="transparent")
        thresh_body.grid(row=1, column=0, columnspan=5, sticky="ew",
                         padx=12, pady=(6, 10))
        thresh_body.columnconfigure((1, 3), weight=1)

        # Confidence
        self.conf_var = ctk.DoubleVar(value=0.25)
        ctk.CTkLabel(thresh_body, text="Confidence:",
                     font=FONT_SMALL, text_color="#bbbbbb").grid(
            row=0, column=0, sticky="w", padx=(0, 6)
        )
        self.conf_lbl = ctk.CTkLabel(thresh_body, text="0.25",
                                      font=FONT_SMALL, width=36)
        self.conf_lbl.grid(row=0, column=1, sticky="w")
        ctk.CTkSlider(
            thresh_body, from_=0.05, to=0.95, number_of_steps=90,
            variable=self.conf_var,
            command=lambda v: self.conf_lbl.configure(text=f"{float(v):.2f}"),
            width=160,
        ).grid(row=0, column=2, padx=(4, 24))

        # IoU
        self.iou_var = ctk.DoubleVar(value=0.45)
        ctk.CTkLabel(thresh_body, text="IoU (NMS):",
                     font=FONT_SMALL, text_color="#bbbbbb").grid(
            row=0, column=3, sticky="w", padx=(0, 6)
        )
        self.iou_lbl = ctk.CTkLabel(thresh_body, text="0.45",
                                     font=FONT_SMALL, width=36)
        self.iou_lbl.grid(row=0, column=4, sticky="w")
        ctk.CTkSlider(
            thresh_body, from_=0.05, to=0.95, number_of_steps=90,
            variable=self.iou_var,
            command=lambda v: self.iou_lbl.configure(text=f"{float(v):.2f}"),
            width=160,
        ).grid(row=0, column=5, padx=(4, 0))

        # Class Filtering & YAML card
        filter_card = ctk.CTkFrame(self, corner_radius=8,
                                   border_width=1, border_color="#3a3a3a")
        filter_card.columnconfigure(1, weight=1)
        filter_card.grid(row=4, column=0, columnspan=3, sticky="ew",
                         padx=8, pady=(0, 8))

        filter_hdr = ctk.CTkFrame(filter_card, fg_color="#2a2a2a", corner_radius=6)
        filter_hdr.grid(row=0, column=0, columnspan=3, sticky="ew")
        ctk.CTkLabel(filter_hdr, text="🎯  Target Classes & Export",
                     font=("Segoe UI Semibold", 12),
                     text_color=ACCENT).grid(row=0, column=0, sticky="w",
                                             padx=12, pady=6)

        filter_body = ctk.CTkFrame(filter_card, fg_color="transparent")
        filter_body.grid(row=1, column=0, columnspan=3, sticky="ew",
                         padx=12, pady=(6, 10))
        filter_body.columnconfigure(1, weight=1)

        ctk.CTkLabel(filter_body, text="Target Classes:",
                     font=FONT_SMALL, text_color="#bbbbbb").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.al_classes_var = ctk.StringVar()
        ctk.CTkEntry(filter_body, textvariable=self.al_classes_var,
                     font=FONT_SMALL,
                     placeholder_text="Additional IDs or names (optional)").grid(
            row=0, column=1, sticky="ew"
        )
        
        self.al_yaml_enabled = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(filter_body, text="Generate data.yaml",
                        variable=self.al_yaml_enabled,
                        font=FONT_SMALL).grid(row=0, column=2, padx=(12, 0))

        # Progress
        self.al_progress = ctk.CTkProgressBar(self)
        self.al_progress.set(0)
        self.al_progress.grid(row=5, column=0, columnspan=3, padx=10,
                              pady=(8, 2), sticky="ew")

        self.al_status = ctk.CTkLabel(self, text="Ready", font=FONT_SMALL,
                                       text_color="gray", wraplength=600, justify="left")
        self.al_status.grid(row=6, column=0, columnspan=3, **PAD, sticky="w")

        # Result stats label
        self.al_result = ctk.CTkLabel(self, text="", font=FONT_SMALL,
                                       text_color="#bbbbbb", justify="left")
        self.al_result.grid(row=7, column=0, columnspan=3, padx=12, pady=(0, 4),
                             sticky="w")

        ctk.CTkButton(self, text="▶  Start Auto Label", font=FONT_LABEL,
                      command=self._start).grid(
            row=8, column=0, columnspan=3, pady=(6, 16)
        )

    def _use_preset(self):
        """Clear custom path when user picks a preset."""
        self.custom_model_var.set("")

    def _toggle_model_type(self, val):
        """Show/Hide rows based on segmented button value."""
        if val == "Preset":
            self.custom_row.grid_forget()
            self.preset_row.grid(row=1, column=0, columnspan=3, sticky="ew")
        else:
            self.preset_row.grid_forget()
            self.custom_row.grid(row=1, column=0, columnspan=3, sticky="ew")

    def _fetch_classes(self):
        """Load model and build a checklist in the scrollable frame."""
        if self.model_type_var.get() == "Preset":
            model_path = self.model_preset_var.get()
        else:
            model_path = self.custom_model_var.get().strip()
            if not model_path:
                messagebox.showerror("Error", "Please select a custom .pt file first.")
                return

        # Pre-check for download status to update UI
        status_msg = "Loading model classes..."
        if self.model_type_var.get() == "Preset":
            if not Path(model_path).exists():
                status_msg = f"Downloading {model_path} (SSL bypass)..."
        
        self.al_status.configure(text=status_msg, text_color="gray")
        
        def _run():
            try:
                names = get_model_classes(model_path)
                
                # Clear frame on main thread
                self.after(0, self._rebuild_checklist, names)
                self.al_status.configure(text="Classes loaded.", text_color="#4CAF50")
            except Exception as exc:
                err_text = str(exc)
                if "SSL" in err_text or "certificate" in err_text.lower():
                    err_text += "\nTIP: If download fails, manually download the file into the project folder."
                self.al_status.configure(text=f"✗ {err_text}", text_color="#F44336")

        threading.Thread(target=_run, daemon=True).start()

    def _rebuild_checklist(self, names):
        """Build checkboxes in the frame (run on main thread)."""
        for widget in self.class_list_frame.winfo_children():
            widget.destroy()
        
        self.class_checkboxes = {}
        row, col = 0, 0
        for idx in sorted(names.keys()):
            name = names[idx]
            cb = ctk.CTkCheckBox(self.class_list_frame, text=f"{idx}: {name}",
                                 font=FONT_SMALL, checkbox_width=18,
                                 checkbox_height=18)
            cb.grid(row=row, column=col, sticky="w", padx=10, pady=(6, 4))
            self.class_checkboxes[idx] = cb
            
            # 3 columns for dense layout
            col += 1
            if col > 2:
                col = 0
                row += 1

    def _start(self):
        inp = self.al_input_var.get().strip()
        out = self.al_output_var.get().strip()
        if not inp or not out:
            messagebox.showerror("Error", "Please select both images and labels folders.")
            return

        if self.model_type_var.get() == "Preset":
            model = self.model_preset_var.get()
        else:
            model = self.custom_model_var.get().strip()
            if not model:
                messagebox.showerror("Error", "Please select a custom .pt file.")
                return

        conf  = round(self.conf_var.get(), 2)
        iou   = round(self.iou_var.get(), 2)

        # Parse target classes (Checkboxes + Manual text)
        target_classes = []
        for cid, cb in self.class_checkboxes.items():
            if cb.get():
                target_classes.append(cid)
        
        raw_manual = self.al_classes_var.get().strip()
        if raw_manual:
            for c in raw_manual.split(","):
                target_classes.append(c.strip())

        # If everything empty, use None (all)
        if not target_classes:
            target_classes = None

        gen_yaml = self.al_yaml_enabled.get()

        self.al_progress.set(0)
        self.al_result.configure(text="")
        self.al_status.configure(text="Loading model…", text_color="gray")

        def _run():
            try:
                def _cb(cur, total):
                    self.al_progress.set(cur / total if total else 0)
                    self.al_status.configure(
                        text=f"Image {cur} / {total}"
                    )

                stats = auto_label(
                    input_folder=inp,
                    output_folder=out,
                    model_name=model,
                    conf_threshold=conf,
                    iou_threshold=iou,
                    target_classes=target_classes, # [ADDED]
                    progress_callback=_cb,
                )

                if gen_yaml:
                    generate_yaml(
                        output_path=str(Path(out) / "data.yaml"),
                        model_names=stats["model_names"]
                    )
                    generate_classes_txt(
                        output_path=str(Path(out) / "classes.txt"),
                        model_names=stats["model_names"]
                    )

                self.al_progress.set(1)
                self.al_status.configure(
                    text=(f"✓  Done – {stats['processed']} images  │  "
                          f"{stats['labeled']} with detections  │  "
                          f"{stats['empty']} empty"),
                    text_color="#4CAF50",
                )

                # Show per-class counts
                cls_lines = "  ".join(
                    f"{name}: {cnt}"
                    for name, cnt in sorted(stats["classes"].items(),
                                            key=lambda x: -x[1])[:10]
                )
                self.al_result.configure(
                    text=f"Top classes → {cls_lines}" if cls_lines else "No detections."
                )

            except Exception as exc:
                self.al_status.configure(text=f"✗  {exc}", text_color="#F44336")

        threading.Thread(target=_run, daemon=True).start()


# ---------------------------------------------------------------------------
# (Tab 4 kept as SplitTab – defined above)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Application window
# ---------------------------------------------------------------------------

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        import sys
        import os
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Data Preparation Tool")
        self.geometry("720x820")
        self.resizable(True, True)
        self.minsize(640, 620)

        # Taskbar icon fix (Windows)
        try:
            myappid = u"antigravity.aidatatool.v1"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass
        
        # Window icon
        icon_path = resource_path("icon.ico")
        if Path(icon_path).exists():
            self.iconbitmap(icon_path)

        # Header
        header = ctk.CTkLabel(
            self,
            text="🗂  AI Data Preparation Tool",
            font=("Segoe UI Semibold", 20),
            text_color=ACCENT,
        )
        header.pack(padx=20, pady=(18, 4), anchor="w")

        ctk.CTkLabel(
            self,
            text="Video  •  Augmentation  •  Auto Label  •  Split Dataset",
            font=FONT_SMALL,
            text_color="gray",
        ).pack(padx=20, anchor="w")

        # Tabview
        tabs = ctk.CTkTabview(self, anchor="nw")
        tabs.pack(fill="both", expand=True, padx=16, pady=12)

        tabs.add("1 · Video → Image")
        tabs.add("2 · Augmentation")
        tabs.add("3 · Auto Label")
        tabs.add("4 · Split Dataset")

        VideoTab(tabs.tab("1 · Video → Image")).pack(fill="both", expand=True)
        AugTab(tabs.tab("2 · Augmentation")).pack(fill="both", expand=True)
        AutoLabelTab(tabs.tab("3 · Auto Label")).pack(fill="both", expand=True)
        SplitTab(tabs.tab("4 · Split Dataset")).pack(fill="both", expand=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Fix SSL certificate error for downloading models
    try:
        import os
        import ssl
        os.environ['PYTHONHTTPSVERIFY'] = '0'
        os.environ['CURL_CA_BUNDLE'] = ''
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass

    app = App()
    app.mainloop()

