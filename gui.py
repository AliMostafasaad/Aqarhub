"""Aqar Hub — GUI v22.6 (Decision Display Fix + Arabic RTL)

Changes:
- Window enlarged to 750x900 for better text display
- wraplength increased to 680 for all labels
- Arabic text gets larger font and right-alignment
- Decision frame gets distinct background color
- Added scrollbar for long reason text
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import pandas as pd
import joblib
import os
import csv
import re
from datetime import datetime
from pathlib import Path

from config import MODEL_PATH, DEBUG_MODE, ALLOWED_TYPES
from data import (_parse_sqm, _parse_payment, _parse_loc, _market_tier,
                  _extract_compound_features, _area_market_zone)
from features import (
    _extract_all_content, _kw_features, _semantic_scores, assemble_X
)
from alert_engine import make_decision, check_down_payment
from type_validator import normalize_property_type, TypeValidationError
from rental_integration import analyze_rental


# ---------- Simple PSI Guard ----------
class PSIGuard:
    def __init__(self, reference_median_ppsm: float):
        self.ref = reference_median_ppsm
        self.samples = []

    def add_query(self, price: float, sqm: float):
        if sqm > 0:
            self.samples.append(price / sqm)
            if len(self.samples) > 100:
                self.samples.pop(0)

    def is_drifted(self) -> bool:
        if len(self.samples) < 20:
            return False
        current_median = sorted(self.samples)[len(self.samples) // 2]
        drift = abs(current_median - self.ref) / self.ref
        return drift > 0.30

    def drift_pct(self) -> float:
        if len(self.samples) < 20:
            return 0.0
        current_median = sorted(self.samples)[len(self.samples) // 2]
        return abs(current_median - self.ref) / self.ref

    def warning_message(self):
        drift = self.drift_pct()
        if drift > 0.40:
            return "⚠️ Market prices appear to have shifted significantly. Estimates may be outdated."
        if drift > 0.25:
            return "ℹ️ Market may have shifted. Use estimates as guidance only."
        return None


# ---------- Main GUI ----------
class AqarHubGUI:
    BUNDLE_KEYS = [
        "model", "tfidf", "svd", "top_feats", "te_maps", "pps_map", "pps_gm",
        "train_medians", "clip_lo", "clip_hi", "comp_counts",
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("Aqar Hub — Price Decision")
        # ── ENLARGED WINDOW ──
        self.root.geometry("750x900")
        self.root.configure(bg="#f5f6fa")

        self.bundle = None
        self.psi_guard = None
        self.load_model()

        self.last_decision = None
        self.last_user_price = None
        self.last_predicted = None
        self.last_prop_type = None

        # ── Load warning icon ──
        self.warning_img = None
        try:
            self.warning_img = tk.PhotoImage(file="download.png")
        except Exception:
            try:
                self.warning_img = tk.PhotoImage(file="warning.png")
            except Exception:
                pass

        self.style = ttk.Style()
        self.style.configure("TFrame", background="#f5f6fa")
        self.style.configure("TLabel", background="#f5f6fa", font=("Segoe UI", 10))
        self.style.configure(
            "Header.TLabel", background="#f5f6fa", font=("Segoe UI", 14, "bold")
        )

        self.build_ui()

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Error", f"Model not found: {MODEL_PATH}")
            self.root.destroy()
            return
        try:
            self.bundle = joblib.load(MODEL_PATH)
            self.validate_bundle()
            ref_ppsm = self.bundle.get("train_median_ppsm", 25000)
            self.psi_guard = PSIGuard(ref_ppsm)
            if DEBUG_MODE:
                print(f"✅ Model loaded, PSI guard active (ref ppsm = {ref_ppsm:.0f})")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.root.destroy()

    def validate_bundle(self):
        missing = [k for k in self.BUNDLE_KEYS if k not in self.bundle]
        if missing:
            raise KeyError(f"Model bundle missing keys: {missing}")

    def build_ui(self):
        # Header
        header = ttk.Label(
            self.root, text="🏠 Aqar Hub", style="Header.TLabel", foreground="#2c3e50"
        )
        header.pack(pady=(15, 5))
        sub = ttk.Label(
            self.root,
            text="Enter property details",
            font=("Segoe UI", 9),
            foreground="#7f8c8d",
        )
        sub.pack(pady=(0, 15))

        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True, padx=20, pady=5)

        # ── MODE SELECTOR ──
        mode_frame = ttk.Frame(container)
        mode_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(mode_frame, text="Analysis Mode:").pack(side="left", padx=(0, 8))
        self.mode_var = tk.StringVar(value="Sale")
        self.mode_combo = ttk.Combobox(
            mode_frame, values=["Sale", "Rental"], textvariable=self.mode_var,
            width=12, state="readonly",
        )
        self.mode_combo.pack(side="left")
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_mode_change)

        # ── RENTAL FEATURES (toggleable) ──
        self.rental_frame = tk.LabelFrame(
            container, text=" Rental Features ", font=("Segoe UI", 10, "bold"),
            bg="#f5f6fa", fg="#2c3e50", padx=10, pady=10,
        )
        self.rental_vars: dict = {}
        rental_feats = [
            ("furnished", "Furnished / مفروش"),
            ("ac", "AC / تكييف"),
            ("parking", "Parking / جراج"),
            ("elevator", "Elevator / مصعد"),
            ("security", "Security / أمن"),
            ("gym", "Gym / جيم"),
            ("pool", "Pool / حمام سباحة"),
            ("balcony", "Balcony / بلكونة"),
            ("garden", "Garden / حديقة"),
            ("storage", "Storage / مخزن"),
            ("generator", "Generator / مولد"),
            ("cameras", "Cameras / كاميرات"),
            ("mosque", "Near Mosque / قرب مسجد"),
            ("school", "Near School / قرب مدرسة"),
            ("hospital", "Near Hospital / قرب مستشفى"),
            ("transport", "Near Transport / مواصلات"),
            ("wifi", "WiFi / إنترنت"),
        ]
        for i, (key, label) in enumerate(rental_feats):
            var = tk.BooleanVar(value=False)
            self.rental_vars[key] = var
            tk.Checkbutton(
                self.rental_frame, text=label, variable=var,
                bg="#f5f6fa", font=("Segoe UI", 9),
            ).grid(row=i // 3, column=i % 3, sticky="w", padx=8, pady=3)


        # ── INPUT FRAME ──
        input_frame = tk.LabelFrame(
            container,
            text=" Property Details ",
            font=("Segoe UI", 10, "bold"),
            bg="#f5f6fa",
            fg="#2c3e50",
            padx=15,
            pady=15,
        )
        input_frame.pack(fill="x", pady=(0, 10))

        # Row 0
        ttk.Label(input_frame, text="Type:").grid(row=0, column=0, sticky="w", pady=5)
        self.type_var = ttk.Combobox(
            input_frame,
            values=sorted(ALLOWED_TYPES),
            width=15,
            state="readonly",
        )
        self.type_var.set("Apartment")
        self.type_var.grid(row=0, column=1, sticky="ew", pady=5, padx=5)

        ttk.Label(input_frame, text="Location (Comp, City):").grid(
            row=0, column=2, sticky="w", pady=5, padx=(15, 0)
        )
        self.location_entry = ttk.Entry(input_frame, width=30)
        self.location_entry.insert(0, "Al Yasmine, New Cairo")
        self.location_entry.grid(row=0, column=3, sticky="ew", pady=5, padx=5)

        # Row 1
        ttk.Label(input_frame, text="Size (sqm):").grid(
            row=1, column=0, sticky="w", pady=5
        )
        self.size_entry = ttk.Entry(input_frame, width=10)
        self.size_entry.insert(0, "120")
        self.size_entry.grid(row=1, column=1, sticky="w", pady=5, padx=5)

        ttk.Label(input_frame, text="Bedrooms:").grid(
            row=1, column=2, sticky="w", pady=5, padx=(15, 0)
        )
        self.bedrooms_entry = ttk.Spinbox(input_frame, from_=0, to=10, width=8)
        self.bedrooms_entry.set(2)
        self.bedrooms_entry.grid(row=1, column=3, sticky="w", pady=5, padx=5)

        # Row 2
        ttk.Label(input_frame, text="Bathrooms:").grid(
            row=2, column=0, sticky="w", pady=5
        )
        self.bathrooms_entry = ttk.Spinbox(input_frame, from_=0, to=10, width=8)
        self.bathrooms_entry.set(1)
        self.bathrooms_entry.grid(row=2, column=1, sticky="w", pady=5, padx=5)

        ttk.Label(input_frame, text="Finish:").grid(
            row=2, column=2, sticky="w", pady=5, padx=(15, 0)
        )
        self.finish_var = ttk.Combobox(
            input_frame,
            values=["Unknown", "Core & Shell", "Semi-Finish", "Fully Finish", "Luxury"],
            width=12,
            state="readonly",
        )
        self.finish_var.set("Fully Finish")
        self.finish_var.grid(row=2, column=3, sticky="w", pady=5, padx=5)

        # Row 3
        ttk.Label(input_frame, text="View:").grid(row=3, column=0, sticky="w", pady=5)
        self.view_var = ttk.Combobox(
            input_frame,
            values=["None", "Pool", "Garden", "Lagoon", "Sea", "Golf"],
            width=10,
            state="readonly",
        )
        self.view_var.set("Garden")
        self.view_var.grid(row=3, column=1, sticky="w", pady=5, padx=5)

        ttk.Label(input_frame, text="Payment:").grid(
            row=3, column=2, sticky="w", pady=5, padx=(15, 0)
        )
        self.payment_var = ttk.Combobox(
            input_frame, values=["Cash", "Installments"], width=10, state="readonly"
        )
        self.payment_var.set("Cash")
        self.payment_var.grid(row=3, column=3, sticky="w", pady=5, padx=5)

        # Row 4
        ttk.Label(input_frame, text="Down Payment (opt):").grid(
            row=4, column=0, sticky="w", pady=5
        )
        self.down_entry = ttk.Entry(input_frame, width=15)
        self.down_entry.grid(row=4, column=1, sticky="w", pady=5, padx=5)

        # Row 5 — Description
        ttk.Label(input_frame, text="Description:").grid(
            row=5, column=0, sticky="nw", pady=5
        )
        self.desc_text = scrolledtext.ScrolledText(
            input_frame, width=60, height=4, font=("Segoe UI", 9), wrap=tk.WORD
        )
        self.desc_text.insert(
            "1.0",
            "Standalone apartment in Al Yasmine residential area, fully finished.",
        )
        self.desc_text.grid(row=5, column=1, columnspan=3, sticky="ew", pady=5, padx=5)

        # ── PRICE FRAME ──
        price_frame = tk.LabelFrame(
            container,
            text=" Your Price ",
            font=("Segoe UI", 10, "bold"),
            bg="#f5f6fa",
            fg="#2c3e50",
            padx=15,
            pady=10,
        )
        price_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(price_frame, text="Asking Price (EGP):").grid(
            row=0, column=0, sticky="w", pady=5
        )
        self.user_price_entry = ttk.Entry(
            price_frame, width=20, font=("Segoe UI", 11)
        )
        self.user_price_entry.insert(0, "5800000")
        self.user_price_entry.grid(row=0, column=1, sticky="w", pady=5, padx=5)

        self.predict_btn = tk.Button(
            price_frame,
            text="🔍 Analyze",
            command=self.analyze,
            bg="#3498db",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            activebackground="#2980b9",
            cursor="hand2",
            padx=20,
            pady=5,
        )
        self.predict_btn.grid(row=0, column=2, padx=(20, 0), pady=5)

        # ── DECISION FRAME (DISTINCT STYLE) ──
        self.result_frame = tk.LabelFrame(
            container,
            text=" Decision ",
            font=("Segoe UI", 11, "bold"),
            bg="#ffffff",
            fg="#2c3e50",
            padx=15,
            pady=15,
            relief="solid",
            bd=2,
        )
        self.result_frame.pack(fill="both", expand=True, pady=(0, 10))

        # Alert row: icon + text side-by-side
        self.alert_frame = tk.Frame(self.result_frame, bg="#ffffff")
        self.alert_frame.pack(pady=(5, 10))

        self.alert_icon = tk.Label(self.alert_frame, bg="#ffffff")
        self.alert_icon.pack(side="left", padx=(0, 15))

        self.alert_label = tk.Label(
            self.alert_frame,
            text="—",
            font=("Segoe UI", 32, "bold"),
            bg="#ffffff",
            fg="#2c3e50",
        )
        self.alert_label.pack(side="left")

        # ── REASON TEXT (SCROLLABLE) ──
        self.reason_container = tk.Frame(self.result_frame, bg="#ffffff")
        self.reason_container.pack(fill="both", expand=True, pady=5)

        self.reason_text = tk.Text(
            self.reason_container,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            bg="#ffffff",
            fg="#2c3e50",
            height=6,
            padx=10,
            pady=10,
            relief="flat",
            state="disabled",
        )
        self.reason_text.pack(side="left", fill="both", expand=True)

        reason_scroll = ttk.Scrollbar(
            self.reason_container, orient="vertical", command=self.reason_text.yview
        )
        reason_scroll.pack(side="right", fill="y")
        self.reason_text.config(yscrollcommand=reason_scroll.set)

        # Down-payment warning
        self.dp_warn_label = tk.Label(
            self.result_frame,
            text="",
            font=("Segoe UI", 9, "bold"),
            bg="#ffffff",
            fg="#e67e22",
            wraplength=680,
            justify="left",
        )
        self.dp_warn_label.pack(pady=(5, 0))

        # Market drift warning
        self.drift_label = tk.Label(
            self.result_frame,
            text="",
            font=("Segoe UI", 9),
            bg="#ffffff",
            fg="#9b59b6",
            wraplength=680,
            justify="left",
        )
        self.drift_label.pack(pady=2)

        # Debug info
        self.debug_label = tk.Label(
            self.result_frame,
            text="",
            font=("Segoe UI", 8),
            bg="#ffffff",
            fg="#7f8c8d",
            wraplength=680,
            justify="left",
        )
        self.debug_label.pack(pady=2)

        # Footer
        footer = tk.Label(
            self.result_frame,
            text="Estimates are based on similar properties and may vary ± typical error.",
            font=("Segoe UI", 8, "italic"),
            bg="#ffffff",
            fg="#7f8c8d",
        )
        footer.pack(pady=(10, 0))

        # ── FEEDBACK ──
        feedback_frame = ttk.Frame(self.result_frame)
        feedback_frame.pack(pady=10)
        ttk.Label(feedback_frame, text="Was this decision correct?").pack(
            side="left", padx=5
        )
        self.btn_correct = tk.Button(
            feedback_frame,
            text="👍 Correct",
            command=lambda: self.save_feedback("up"),
            bg="#2ecc71",
            fg="white",
            font=("Segoe UI", 9),
            padx=10,
            state="disabled",
        )
        self.btn_correct.pack(side="left", padx=5)
        self.btn_incorrect = tk.Button(
            feedback_frame,
            text="👎 Incorrect",
            command=lambda: self.save_feedback("down"),
            bg="#e74c3c",
            fg="white",
            font=("Segoe UI", 9),
            padx=10,
            state="disabled",
        )
        self.btn_incorrect.pack(side="left", padx=5)

        self.status = ttk.Label(
            self.root,
            text="Ready",
            relief="sunken",
            anchor="w",
            font=("Segoe UI", 9),
            foreground="#7f8c8d",
        )
        self.status.pack(fill="x", side="bottom", pady=(5, 0))

    def save_feedback(self, value):
        if self.last_decision is None:
            return
        path = Path("user_feedback.csv")
        exists = path.exists()
        try:
            with open(path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not exists:
                    writer.writerow(
                        [
                            "timestamp",
                            "price",
                            "predicted",
                            "type",
                            "alert",
                            "confidence",
                            "feedback",
                        ]
                    )
                writer.writerow(
                    [
                        datetime.now().isoformat(),
                        float(self.last_user_price),
                        float(self.last_predicted),
                        self.last_prop_type,
                        self.last_decision.get("alert"),
                        self.last_decision.get("confidence"),
                        value,
                    ]
                )
            if DEBUG_MODE:
                print(f"📝 Feedback saved: {value}")
            messagebox.showinfo("Thank You", "Feedback recorded.")
        except Exception as e:
            if DEBUG_MODE:
                print(f"Error saving feedback: {e}")

    def get_finish_ord(self):
        mapping = {
            "Unknown": np.nan,
            "Core & Shell": 0.0,
            "Semi-Finish": 1.0,
            "Fully Finish": 2.0,
            "Luxury": 3.0,
        }
        return mapping.get(self.finish_var.get(), np.nan)

    def get_view_score(self):
        mapping = {
            "None": 0,
            "Pool": 1,
            "Garden": 2,
            "Lagoon": 3,
            "Sea": 4,
            "Golf": 5,
        }
        return mapping.get(self.view_var.get(), 0)

    def build_dataframe(self):
        loc_str = self.location_entry.get()
        loc = _parse_loc(loc_str)
        compound = loc["compound"]
        city = loc["city"]
        governorate = loc["governorate"]

        try:
            size_sqm = float(self.size_entry.get())
        except ValueError:
            size_sqm = np.nan

        bedrooms = float(self.bedrooms_entry.get())
        bathrooms = float(self.bathrooms_entry.get())
        finish_ord = self.get_finish_ord()
        view_score = self.get_view_score()
        desc = self.desc_text.get("1.0", tk.END).strip()
        payment = self.payment_var.get().lower()
        is_installment = 1 if "install" in payment else 0
        is_cash = 1 if "cash" in payment else 0

        dp_str = self.down_entry.get().strip()
        down_payment = _parse_payment(dp_str) if dp_str else np.nan

        market_tier = _market_tier(governorate, city, compound)
        comp_counts = self.bundle.get("comp_counts", {})
        log_comp_size = np.log1p(comp_counts.get(compound, 10))

        room_ratio = bathrooms / bedrooms if bedrooms > 0 else np.nan
        size_per_bedroom = size_sqm / bedrooms if bedrooms > 0 else np.nan
        size_x_finish = (
            (size_sqm if pd.notna(size_sqm) else 0)
            * (finish_ord if pd.notna(finish_ord) else 0)
        )

        yr_match = re.search(r"(\d+)\s*year", desc.lower())
        install_years = float(yr_match.group(1)) if yr_match else np.nan

        user_price = float(self.user_price_entry.get().replace(",", ""))
        price_egp = user_price

        down_payment_ratio = np.nan
        log_down_payment = np.nan
        if pd.notna(down_payment) and price_egp > 0:
            dpr = down_payment / price_egp
            if 0 < dpr < 1.0:
                down_payment_ratio = dpr
                log_down_payment = np.log1p(down_payment)

        url = "https://example.com/plp/buy/"
        raw_type = self.type_var.get().strip()

        try:
            model_type = normalize_property_type(raw_type)
        except TypeValidationError:
            model_type = raw_type.title()

        row = {
            "url": url,
            "price": str(price_egp),
            "price_egp": price_egp,
            "log_price": np.log1p(price_egp),
            "description": desc,
            "desc_clean": desc,
            "location": loc_str,
            "type": raw_type,
            "type_clean": model_type,
            "size": str(size_sqm) if not np.isnan(size_sqm) else "",
            "size_sqm": size_sqm,
            "bedrooms": str(bedrooms),
            "bedrooms_n": bedrooms,
            "bathrooms": str(bathrooms),
            "bathrooms_n": bathrooms,
            "has_maid": 1 if "maid" in desc.lower() else 0,
            "payment_method": payment,
            "is_installment": is_installment,
            "is_cash": is_cash,
            "down_payment": dp_str if dp_str else "",
            "down_payment_ratio": down_payment_ratio,
            "log_down_payment": log_down_payment,
            "room_ratio": room_ratio,
            "compound": compound,
            "city": city,
            "governorate": governorate,
            "install_years": install_years,
            "log_comp_size": log_comp_size,
            "finish_ord": finish_ord,
            "view_score": view_score,
            "market_tier": market_tier,
            "size_per_bedroom": size_per_bedroom,
            "size_x_finish": size_x_finish,
            "log_size_sqm": np.log1p(size_sqm) if pd.notna(size_sqm) and size_sqm > 0 else np.log1p(1),
            "size_sqm_inv": (1.0 / size_sqm) if pd.notna(size_sqm) and size_sqm > 0 else 1.0,
            "area_market_zone": _area_market_zone(city, compound, desc),
        }

        df = pd.DataFrame([row])
        df = _extract_compound_features(df)
        df["city_comp"] = df["city"] + "_comp" + df["is_compound"].astype(str)
        df["governorate_comp"] = df["governorate"] + "_comp" + df["is_compound"].astype(str)

        if DEBUG_MODE:
            print(f"🔍 INFERENCE CHECK: is_compound={df['is_compound'].iloc[0]}, "
                  f"compound_tier={df['compound_tier'].iloc[0]}, "
                  f"standalone_premium={df['standalone_premium'].iloc[0]}, "
                  f"city_comp={df['city_comp'].iloc[0]}")

        return df

    def preprocess(self, df):
        bundle = self.bundle
        tfidf = bundle["tfidf"]
        svd = bundle["svd"]
        te_maps = bundle["te_maps"]
        pps_map = bundle["pps_map"]
        pps_gm = bundle["pps_gm"]
        train_medians = bundle["train_medians"]
        lo = bundle["clip_lo"]
        hi = bundle["clip_hi"]
        top_feats = bundle["top_feats"]

        v20 = _extract_all_content(df["desc_clean"])
        df = pd.concat([df.reset_index(drop=True), v20.reset_index(drop=True)], axis=1)

        df["price_per_sqm_estimate"] = df[["city", "type_clean"]].apply(
            lambda r: pps_map.get((r["city"], r["type_clean"]), pps_gm), axis=1
        )

        for col in ["type_clean", "governorate", "city", "compound"]:
            m = te_maps[col]["map"]
            gm = te_maps[col]["gm"]
            df[col + "_te"] = df[col].map(m).fillna(gm)

        if "city_comp" in te_maps:
            m = te_maps["city_comp"]["map"]
            gm = te_maps["city_comp"]["gm"]
            df["city_comp_te"] = df["city_comp"].map(m).fillna(gm)
        else:
            df["city_comp_te"] = te_maps.get("type_clean", {}).get("gm", 0.0)

        if "governorate_comp" in te_maps:
            m = te_maps["governorate_comp"]["map"]
            gm = te_maps["governorate_comp"]["gm"]
            df["governorate_comp_te"] = df["governorate_comp"].map(m).fillna(gm)
        else:
            df["governorate_comp_te"] = te_maps.get("type_clean", {}).get("gm", 0.0)

        df["comp_type"] = df["compound"] + "_" + df["type_clean"]
        df["bed_type"] = (
            df["bedrooms_n"].fillna(0).astype(int).astype(str) + "_" + df["type_clean"]
        )
        d2 = df["desc_clean"].str.lower()
        df["is_nc"] = d2.str.contains(
            r"north\s*coast|ras\s*el?\s*hekma|sahel", regex=True
        ).astype(np.int8)
        df["nc_type"] = df["is_nc"].astype(str) + "_" + df["type_clean"]
        df["finish_type"] = (
            df["finish_ord"].fillna(-1).astype(int).astype(str) + "_" + df["type_clean"]
        )
        df["type_tier"] = (
            df["type_clean"] + "_" + df["market_tier"].astype(int).astype(str)
        )

        for combo in ["comp_type", "bed_type", "nc_type", "finish_type", "type_tier"]:
            m = te_maps[combo]["map"]
            gm = te_maps[combo]["gm"]
            df[combo + "_te"] = df[combo].map(m).fillna(gm)

        kw = _kw_features(df["desc_clean"])
        sem = _semantic_scores(df["desc_clean"])
        for c in ["luxury_score", "budget_score", "investment_signal"]:
            df[c] = sem[c].values

        mat = tfidf.transform(df["desc_clean"])
        lsa = svd.transform(mat).astype(np.float32)
        # ── DYNAMIC LSA COLUMN GENERATION ──
        # Uses actual SVD component count from loaded model bundle instead of hardcoded 25
        lsa_df = pd.DataFrame(
            lsa, columns=[f"lsa_{i}" for i in range(lsa.shape[1])], index=df.index
        )
        X = assemble_X(df, kw, lsa_df, train_medians=train_medians, lo=lo, hi=hi)
        return X[top_feats], df

    def analyze(self):
        if self.bundle is None:
            messagebox.showerror("Error", "Model not loaded!")
            return

        if self.mode_var.get() == "Rental":
            self._analyze_rental()
        else:
            self._analyze_sale()

    def _on_mode_change(self, event=None):
        if self.mode_var.get() == "Rental":
            self.rental_frame.pack(fill="x", pady=(0, 10), after=self.input_frame)
            self.price_frame.config(text=" Desired Monthly Rent (EGP) ")
            self.user_price_entry.delete(0, tk.END)
            self.user_price_entry.insert(0, "25000")
        else:
            self.rental_frame.pack_forget()
            self.price_frame.config(text=" Your Price ")
            self.user_price_entry.delete(0, tk.END)
            self.user_price_entry.insert(0, "5800000")

    def _analyze_sale(self):
        if self.bundle is None:
            messagebox.showerror("Error", "Model not loaded!")
            return

        # Reset
        self.dp_warn_label.config(text="")
        self.drift_label.config(text="")
        self.debug_label.config(text="")
        self.alert_icon.config(image="")
        self._set_reason_text("")

        try:
            self.status.config(text="Processing...", foreground="#e67e22")
            self.root.update()

            raw_type = self.type_var.get().strip()

            try:
                normalized_type = normalize_property_type(raw_type)
            except TypeValidationError:
                messagebox.showerror(
                    "Unsupported Type",
                    f"'{raw_type}' is not supported.\n"
                    f"Allowed: {', '.join(sorted(ALLOWED_TYPES))}\n\n"
                    f"Note: Villa variants (iVilla, Twin Villa, etc.) are normalized to 'Villa'.",
                )
                self.status.config(text="Ready", foreground="#7f8c8d")
                return

            try:
                user_price = float(self.user_price_entry.get().replace(",", ""))
                if user_price < 10_000 or user_price > 500_000_000:
                    messagebox.showerror(
                        "Invalid Price", "Price must be between 10,000 and 500M EGP."
                    )
                    self.status.config(text="Ready", foreground="#7f8c8d")
                    return
            except ValueError:
                messagebox.showerror("Invalid Price", "Enter a valid number.")
                self.status.config(text="Ready", foreground="#7f8c8d")
                return

            if not self.location_entry.get().strip():
                messagebox.showwarning("Missing Location", "Please enter a location.")
                self.status.config(text="Ready", foreground="#7f8c8d")
                return

            try:
                size_sqm = float(self.size_entry.get())
                if size_sqm < 5 or size_sqm > 5000:
                    raise ValueError
            except ValueError:
                messagebox.showwarning(
                    "Invalid Size", "Enter size in sqm (e.g., 120)."
                )
                self.status.config(text="Ready", foreground="#7f8c8d")
                return

            # Build and predict
            df_input = self.build_dataframe()
            X, df_proc = self.preprocess(df_input)
            model = self.bundle["model"]
            lo_price, med_price, hi_price = model.predict_price_interval(X)

            if med_price[0] <= 0:
                messagebox.showerror("Model Error", "Prediction failed.")
                self.status.config(text="Error", foreground="#e74c3c")
                return

            # Down-payment detection
            dp_warning = check_down_payment(user_price, med_price[0])
            if dp_warning:
                self.dp_warn_label.config(text=dp_warning)
                if DEBUG_MODE:
                    print(f"⚠️ {dp_warning}")

            # PSI / Market drift guard
            self.psi_guard.add_query(user_price, size_sqm)
            drift_msg = self.psi_guard.warning_message()
            if drift_msg:
                self.drift_label.config(text=drift_msg)
                if DEBUG_MODE:
                    print(drift_msg)
            if self.psi_guard.is_drifted() and DEBUG_MODE:
                print("⚠️ PSI WARNING: Market drift detected (>30% change in price per sqm).")

            # Area for decision engine
            area = df_proc.iloc[0].get("city", "Unknown")
            if area == "Unknown":
                area = self.location_entry.get().split(",")[-1].strip() or "Unknown"

            desc_lower = self.desc_text.get("1.0", tk.END).lower()
            has_pool = "pool" in desc_lower or "swimming" in desc_lower
            premium_devs = [
                "palm hills", "sodic", "orascom", "emaar",
                "mountain view", "la vista", "hyde park",
            ]
            developer_level = (
                "premium" if any(d in desc_lower for d in premium_devs) else "normal"
            )
            if developer_level == "normal" and "developer" not in desc_lower:
                developer_level = "unknown"

            # Extract is_compound from processed dataframe
            is_compound_flag = df_proc.iloc[0].get("is_compound", 0) == 1
            compound_val = df_proc.iloc[0].get("compound", "")
            desc_val = self.desc_text.get("1.0", tk.END).strip()

            features = {
                "land_size": "present"
                if pd.notna(df_input.iloc[0].get("size_sqm"))
                else "missing",
                "in_compound": "yes" if is_compound_flag else "no",
                "has_pool": "yes" if has_pool else "no",
                "developer_level": developer_level,
                "desc": desc_val,
                "compound": compound_val,
            }

            decision = make_decision(
                actual_price=user_price,
                predicted_price=med_price[0],
                lo_price=lo_price[0],
                hi_price=hi_price[0],
                property_type=normalized_type,
                area=area,
                features=features,
                size_sqm=size_sqm if pd.notna(size_sqm) else None,
                bundle=self.bundle,
                payment_type=self.payment_var.get().lower(),
                installment_years=df_input.iloc[0].get("install_years"),
                down_payment_ratio=df_input.iloc[0].get("down_payment_ratio"),
            )

            self.last_decision = decision
            self.last_user_price = user_price
            self.last_predicted = med_price[0]
            self.last_prop_type = normalized_type

            self.btn_correct.config(state="normal")
            self.btn_incorrect.config(state="normal")

            # Display
            alert = decision["alert"]
            if alert == "UNDERPRICED":
                alert_text = "UNDERPRICED 🟢"
                color = "#27ae60"
            elif alert == "OVERPRICED":
                alert_text = "OVERPRICED 🔴"
                color = "#e74c3c"
            elif alert == "ESTIMATE ONLY":
                alert_text = "ESTIMATE ONLY 🟡"
                color = "#f39c12"
            else:
                alert_text = "FAIR 🟠"
                color = "#f39c12"

            self.alert_label.config(text=alert_text, fg=color)
            self._set_reason_text(decision["reason"])

            # Show warning icon beside alert
            if self.warning_img:
                self.alert_icon.config(image=self.warning_img)

            # Show inference verification
            is_comp = df_proc.iloc[0].get("is_compound", -1)
            comp_tier = df_proc.iloc[0].get("compound_tier", -1)
            st_prem = df_proc.iloc[0].get("standalone_premium", -1)
            debug_text = f"is_compound={int(is_comp)} | tier={comp_tier} | standalone_premium={int(st_prem)}"
            self.debug_label.config(text=debug_text)

            self.status.config(text="Analysis complete", foreground="#27ae60")
            if DEBUG_MODE:
                print("Decision:", decision)
                print(f"🔍 Inference signals: {debug_text}")

        except Exception as e:
            self.status.config(text="Error", foreground="#e74c3c")
            messagebox.showerror("Analysis Error", str(e))
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()

    def _analyze_rental(self):
        try:
            self.status.config(text="Processing rental...", foreground="#e67e22")
            self.root.update()

            self.dp_warn_label.config(text="")
            self.drift_label.config(text="")
            self.debug_label.config(text="")
            self.alert_icon.config(image="")
            self._set_reason_text("")

            # --- validations (mirror sale logic) ---
            raw_type = self.type_var.get().strip()
            try:
                normalized_type = normalize_property_type(raw_type)
            except TypeValidationError:
                messagebox.showerror(
                    "Unsupported Type",
                    f"'{raw_type}' not supported.\nAllowed: {', '.join(sorted(ALLOWED_TYPES))}"
                )
                self.status.config(text="Ready", foreground="#7f8c8d")
                return

            try:
                user_rent = float(self.user_price_entry.get().replace(",", ""))
                if user_rent < 500 or user_rent > 500_000:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid Rent", "Monthly rent must be 500 – 500,000 EGP.")
                self.status.config(text="Ready", foreground="#7f8c8d")
                return

            if not self.location_entry.get().strip():
                messagebox.showwarning("Missing Location", "Please enter a location.")
                self.status.config(text="Ready", foreground="#7f8c8d")
                return

            try:
                size_sqm = float(self.size_entry.get())
                if size_sqm < 5 or size_sqm > 5000:
                    raise ValueError
            except ValueError:
                messagebox.showwarning("Invalid Size", "Enter size in sqm.")
                self.status.config(text="Ready", foreground="#7f8c8d")
                return

            # --- reuse existing sale pipeline for base valuation ---
            df_input = self.build_dataframe()
            X, df_proc = self.preprocess(df_input)

            checkbox_features = {k: v.get() for k, v in self.rental_vars.items()}

            result = analyze_rental(
                user_rent=user_rent,
                df_proc=df_proc,
                X=X,
                bundle=self.bundle,
                checkbox_features=checkbox_features,
                desc_text=self.desc_text.get("1.0", tk.END).strip(),
            )

            # --- display ---
            alert = result["alert"]
            if alert == "UNDERPRICED":
                text, color = "UNDERPRICED 🟢", "#27ae60"
            elif alert == "OVERPRICED":
                text, color = "OVERPRICED 🔴", "#e74c3c"
            else:
                text, color = "FAIR 🟠", "#f39c12"

            self.alert_label.config(text=text, fg=color)
            self._set_reason_text(result["reason"])

            dbg = (
                f"Sale est: {result['sale_price_estimate']/1e6:.2f}M | "
                f"Yield: {result['yield']['typical']*100:.1f}% | "
                f"Premium: +{result['premium_pct']*100:.0f}% | "
                f"Features: {result['feature_score']} pts"
            )
            self.debug_label.config(text=dbg)

            self.last_decision = result
            self.last_user_price = result["user_rent"]
            self.last_predicted = result["expected_rent_range"]["mid"]
            self.last_prop_type = result["mode"]
            self.btn_correct.config(state="normal")
            self.btn_incorrect.config(state="normal")

            self.status.config(text="Rental analysis complete", foreground="#27ae60")

        except Exception as e:
            self.status.config(text="Error", foreground="#e74c3c")
            messagebox.showerror("Rental Analysis Error", str(e))
            if DEBUG_MODE:
                import traceback
                traceback.print_exc()

    def _set_reason_text(self, text: str):
        """Helper to update the scrollable reason text widget."""
        self.reason_text.config(state="normal")
        self.reason_text.delete("1.0", tk.END)
        self.reason_text.insert("1.0", text)
        self.reason_text.config(state="disabled")


def main():
    root = tk.Tk()
    app = AqarHubGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()