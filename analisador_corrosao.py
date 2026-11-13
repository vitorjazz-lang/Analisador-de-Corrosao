import csv
import json
import math
import os
import queue
import threading
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk


APP_TITLE = "Analisador de Corrosão"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Cores do overlay (BGR)
COR_FERRUGEM = (0, 140, 255)   # laranja
COR_ESCURA = (255, 0, 200)     # magenta
COR_NUM = (0, 0, 255)          # vermelho
COR_CHAPA = (0, 200, 0)        # verde


# ----------------------------------------------------------------------------
# 1. Localização da chapa
# ----------------------------------------------------------------------------
def find_plate(bgr):
    """Retorna máscara booleana da chapa (maior objeto diferente do fundo)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hist = np.bincount(gray.ravel(), minlength=256)
    bg_gray = 180 + int(hist[180:].argmax())
    bg_pix = bgr[np.abs(gray.astype(int) - bg_gray) < 2]
    bg = np.median(bg_pix, axis=0) if len(bg_pix) else np.array([247, 247, 247])

    diff = np.abs(bgr.astype(np.float32) - bg).max(axis=2)
    mask = (diff > 7).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = ndi.binary_fill_holes(mask).astype(np.uint8)

    ks = max(15, int(min(mask.shape) * 0.10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((ks, ks), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        raise RuntimeError("Não foi possível localizar a chapa.")
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    plate = lab == biggest
    return ndi.binary_fill_holes(plate)


def inner_region(plate, margin_frac=0.025):
    """Afasta da borda da chapa (borda escura/chanfro não é corrosão)."""
    ys, xs = np.where(plate)
    if len(ys) == 0:
        raise RuntimeError("Máscara da chapa vazia.")
    side = min(ys.max() - ys.min(), xs.max() - xs.min())
    m = max(3, int(side * margin_frac))
    return cv2.erode(plate.astype(np.uint8), np.ones((2 * m + 1, 2 * m + 1), np.uint8)).astype(bool)


# ----------------------------------------------------------------------------
# 2. Marcas a ignorar
# ----------------------------------------------------------------------------
def find_marks(lab_a, lab_b, L, region):
    """Pontos vermelhos de identificação e manchas pretas/furos."""
    chroma = np.hypot(lab_a, lab_b)
    red = (lab_a > 30) & (chroma > 38)
    black = L < 22
    marks = (red | black) & region
    marks = cv2.morphologyEx(marks.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(marks, 8)
    out = np.zeros_like(marks, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 25:
            out |= lab == i
    out = cv2.dilate(out.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))).astype(bool)
    return out


def count_red_dots(bgr):
    """Conta pontos vermelhos na imagem inteira (miniatura de ID ou sobre a chapa)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    red = ((a > 30) & (np.hypot(a, b) > 38)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(red, 8)
    return int(sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 25))


# ----------------------------------------------------------------------------
# 3. Segmentação da corrosão
# ----------------------------------------------------------------------------
def segment_corrosion(bgr, plate, sens=1.0, ref_bottom=0.08):
    """Retorna (rust_mask, dark_mask, analysis_region, info)."""
    if sens <= 0:
        raise ValueError("A sensibilidade precisa ser maior que zero.")

    region = inner_region(plate)

    sm = cv2.bilateralFilter(bgr, 9, 30, 9)
    sm = cv2.GaussianBlur(sm, (0, 0), 1.2)
    lab = cv2.cvtColor(sm, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * 100.0 / 255.0
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0

    marks = find_marks(a, b, L, region)
    region = region & ~marks

    ys = np.where(region)[0]
    if len(ys) == 0:
        raise RuntimeError("Área insuficiente para análise após remover marcas.")

    y0, y1 = ys.min(), ys.max()
    h = y1 - y0
    strip = np.zeros_like(region)
    strip[int(y1 - h * ref_bottom):y1, :] = True
    strip &= region
    cand = strip if strip.sum() > 500 else region
    b_ref = np.percentile(b[cand], 30)
    a_ref = np.percentile(a[cand], 30)

    da, db = a - a_ref, b - b_ref
    rust_score = 0.35 * da + 0.94 * db
    vals = rust_score[region]
    v8 = np.clip((vals - vals.min()) / (np.ptp(vals) + 1e-6) * 255, 0, 255).astype(np.uint8)
    t_otsu, _ = cv2.threshold(v8.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t_otsu = vals.min() + t_otsu / 255.0 * np.ptp(vals)
    t_rust = float(np.clip(t_otsu, 6.0, 14.0)) / sens
    rust = (rust_score > t_rust) & region

    L8 = (L * 255 / 100).astype(np.uint8)
    L8m = L8.copy()
    L8m[~region] = np.uint8(np.percentile(L8[region], 90))
    ksz = max(31, (int(min(bgr.shape[:2]) * 0.06) // 2) * 2 + 1)
    bg_L = cv2.morphologyEx(L8m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)))
    bg_L = cv2.GaussianBlur(bg_L, (0, 0), ksz / 3).astype(np.float32) * 100.0 / 255.0
    L_s = cv2.GaussianBlur(L, (0, 0), 2.5)
    drop = bg_L - L_s
    chroma_dev = np.hypot(da, db)
    t_dark = 20.0 / sens

    zc = max(15, int(min(bgr.shape[:2]) * 0.02)) | 1
    zone = cv2.morphologyEx(rust.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (zc * 2 + 1, zc * 2 + 1)))
    zone = cv2.dilate(zone, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (zc, zc))).astype(bool)
    dark = (drop > t_dark) & (chroma_dev < t_rust * 1.6) & region & ~rust & zone

    return rust, dark, region, dict(b_ref=float(b_ref), a_ref=float(a_ref),
                                    t_rust=float(t_rust), t_dark=float(t_dark))


def clean_mask(mask, min_area=12, min_minor_axis=2.8):
    """Remove ruído e riscos finos."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    lab = measure.label(m, connectivity=2)
    out = np.zeros_like(m, bool)
    for r in measure.regionprops(lab):
        if r.area < min_area:
            continue
        if r.axis_minor_length < min_minor_axis and r.axis_major_length > 8 * max(r.axis_minor_length, 1):
            continue
        out[lab == r.label] = True
    return out


# ----------------------------------------------------------------------------
# 4. Métricas e referência ASTM D610 (aproximada)
# ----------------------------------------------------------------------------
def astm_d610_grade(pct):
    limits = [(0.01, 10), (0.03, 9), (0.1, 8), (0.3, 7), (1, 6), (3, 5),
              (10, 4), (16, 3), (33, 2), (50, 1)]
    for lim, g in limits:
        if pct <= lim:
            return g
    return 0


def spots_table(mask, tag, px_per_mm=None):
    lab = measure.label(mask, connectivity=2)
    rows = []
    for r in measure.regionprops(lab):
        cy, cx = r.centroid
        d = r.equivalent_diameter_area
        row = dict(tipo=tag, x=int(cx), y=int(cy), area_px=int(r.area),
                   diametro_eq_px=round(float(d), 1), bbox=r.bbox, label=r.label)
        if px_per_mm:
            row["area_mm2"] = round(r.area / px_per_mm ** 2, 3)
            row["diametro_eq_mm"] = round(d / px_per_mm, 3)
        rows.append(row)
    return rows, lab


# ----------------------------------------------------------------------------
# 5. Overlay
# ----------------------------------------------------------------------------
def draw_overlay(bgr, rust, dark, plate, rows_all, n_label=40):
    out = bgr.copy()
    tint = out.copy()
    tint[rust] = COR_FERRUGEM
    tint[dark] = COR_ESCURA
    out = cv2.addWeighted(tint, 0.45, out, 0.55, 0)

    for m, col in ((rust, COR_FERRUGEM), (dark, COR_ESCURA)):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, col, 1, cv2.LINE_AA)

    pc, _ = cv2.findContours(plate.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, pc, -1, COR_CHAPA, 2, cv2.LINE_AA)

    big = sorted(rows_all, key=lambda r: -r["area_px"])[:n_label]
    for idx, r in enumerate(big, 1):
        x, y = r["x"], r["y"]
        cv2.putText(out, str(idx), (x - 6, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, str(idx), (x - 6, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, COR_NUM, 1, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------------
# 6. Pipeline por imagem
# ----------------------------------------------------------------------------
def analyze(path, outdir, sens=1.0, min_area=12, px_per_mm=None, n_label=40,
            output_stem=None):
    if px_per_mm is not None:
        px_per_mm = float(px_per_mm)
        if not math.isfinite(px_per_mm) or px_per_mm <= 0:
            raise ValueError("A calibração precisa ser um valor positivo em px/mm.")

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"Não consegui abrir {path}")
    stem = output_stem or Path(path).stem

    plate = find_plate(bgr)
    rust, dark, region, info = segment_corrosion(bgr, plate, sens=sens)
    rust = clean_mask(rust, min_area)
    dark = clean_mask(dark, max(min_area, 20), min_minor_axis=3.5)
    dark &= ~rust

    area_region = int(region.sum())
    if area_region <= 0:
        raise RuntimeError("Área de análise vazia.")

    pct_rust = 100.0 * rust.sum() / area_region
    pct_dark = 100.0 * dark.sum() / area_region
    pct_total = 100.0 * (rust | dark).sum() / area_region

    rows_r, _ = spots_table(rust, "ferrugem", px_per_mm)
    rows_d, _ = spots_table(dark, "escura", px_per_mm)
    rows_all = rows_r + rows_d
    rows_all.sort(key=lambda r: -r["area_px"])
    for i, r in enumerate(rows_all, 1):
        r["id"] = i

    # Guarda APENAS a imagem principal (overlay)
    overlay = draw_overlay(bgr, rust, dark, plate, rows_all, n_label)
    overlay_path = outdir / f"{stem}_overlay.png"
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"Não consegui salvar a imagem de resultado: {overlay_path}")

    # As linhas que guardavam _mascara.png, _pontos.csv e _relatorio.json 
    # foram removidas daqui para não poluir a pasta.

    resumo = dict(
        arquivo=Path(path).name,
        pontos_vermelhos_id=count_red_dots(bgr),
        area_analisada_px=area_region,
        pct_ferrugem=round(pct_rust, 2),
        pct_escura=round(pct_dark, 2),
        pct_corrosao_total=round(pct_total, 2),
        nota_astm_d610_aprox=astm_d610_grade(pct_total),
        n_manchas_ferrugem=len(rows_r),
        n_manchas_escuras=len(rows_d),
        n_manchas_total=len(rows_all),
        maior_mancha_px=max([r["area_px"] for r in rows_all], default=0),
        limiar_ferrugem=round(info["t_rust"], 2),
        limiar_escura=round(info["t_dark"], 2),
    )
    if px_per_mm is not None:
        resumo.update(
            px_por_mm=round(px_per_mm, 6),
            area_analisada_mm2=round(area_region / px_per_mm ** 2, 3),
            area_corrosao_total_mm2=round((rust | dark).sum() / px_per_mm ** 2, 3),
            maior_mancha_mm2=round(resumo["maior_mancha_px"] / px_per_mm ** 2, 3),
        )

    return resumo, overlay_path


def parse_px_per_mm(value):
    """Converte a calibração opcional digitada na interface."""
    value = value.strip().replace(",", ".")
    if not value:
        return None
    try:
        px_per_mm = float(value)
    except ValueError as exc:
        raise ValueError("Informe a calibração como um número em px/mm.") from exc
    if not math.isfinite(px_per_mm) or px_per_mm <= 0:
        raise ValueError("A calibração precisa ser um valor positivo em px/mm.")
    return px_per_mm


def output_stems(paths):
    """Gera nomes de saída distintos quando as imagens têm o mesmo nome-base."""
    stems = [Path(path).stem for path in paths]
    counts = {}
    for stem in stems:
        counts[stem.casefold()] = counts.get(stem.casefold(), 0) + 1

    reserved = set(counts)
    allocated = set()
    result = []
    for path, stem in zip(paths, stems):
        candidate = stem
        if counts[stem.casefold()] > 1:
            candidate = f"{stem}_{Path(path).parent.name}"
        base = candidate
        suffix = 2
        while candidate.casefold() in allocated or (
            counts[stem.casefold()] > 1 and candidate.casefold() in reserved
        ):
            candidate = f"{base}_{suffix}"
            suffix += 1
        allocated.add(candidate.casefold())
        result.append(candidate)
    return result


def save_summary(results, output_dir):
    if not results:
        return None
    csv_path = output_dir / "resumo.csv"
    json_path = output_dir / "resumo.json"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        fields = list(results[0].keys())
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    return csv_path


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1200x760")
        self.minsize(1000, 680)

        self.files = []
        self.current_photo = None
        self.analysis_queue = queue.Queue()
        self.analysis_thread = None

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        self.select_button = ttk.Button(top, text="Selecionar imagens", command=self.select_images)
        self.select_button.pack(
            side="left", padx=4
        )
        self.analyze_button = ttk.Button(top, text="Analisar", command=self.run_analysis)
        self.analyze_button.pack(
            side="left", padx=4
        )
        ttk.Button(top, text="Abrir pasta de saída", command=self.open_output).pack(
            side="left", padx=4
        )

        ttk.Label(top, text="Sensibilidade:").pack(side="left", padx=(18, 5))

        self.sens_var = tk.DoubleVar(value=1.0)
        self.sens_scale = ttk.Scale(
            top,
            from_=0.60,
            to=1.35,
            variable=self.sens_var,
            orient="horizontal",
            length=220,
        )
        self.sens_scale.pack(side="left")

        self.sens_label = ttk.Label(top, text="1.00")
        self.sens_label.pack(side="left", padx=6)
        self.sens_var.trace_add("write", self.update_sens_label)

        calibration = ttk.Frame(self, padding=(10, 0))
        calibration.pack(fill="x", pady=(0, 6))
        ttk.Label(calibration, text="Calibração em px/mm (opcional):").pack(side="left")
        self.calibration_var = tk.StringVar()
        ttk.Entry(calibration, textvariable=self.calibration_var, width=12).pack(
            side="left", padx=6
        )
        ttk.Label(calibration, text="Aplicada a todas as imagens deste lote.").pack(
            side="left", padx=6
        )

        self.listbox = tk.Listbox(self, height=6)
        self.listbox.pack(fill="x", padx=10, pady=(0, 10))

        main = ttk.Panedwindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=10, pady=5)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=4)
        main.add(right, weight=2)

        self.image_label = ttk.Label(
            left,
            text="Nenhuma imagem analisada",
            anchor="center",
            relief="sunken",
        )
        self.image_label.pack(fill="both", expand=True, padx=5, pady=5)

        ttk.Label(
            right,
            text=(
                "RESULTADO\\n\\n"
                "Ferrugem aparece em laranja e regiões escuras associadas em magenta.\\n"
                "São salvos o overlay e os resumos CSV/JSON.\\n"
                "A calibração opcional habilita áreas em mm²."
            ),
            justify="left",
        ).pack(anchor="w", padx=10, pady=10)

        self.stats = tk.Text(right, height=16, width=42)
        self.stats.pack(fill="both", expand=True, padx=10, pady=10)

        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x", side="bottom")

        self.status = ttk.Label(self, text="Pronto.", relief="sunken", anchor="w")
        self.status.pack(fill="x", side="bottom")

    def update_sens_label(self, *_args):
        self.sens_label.config(text=f"{self.sens_var.get():.2f}")

    def select_images(self):
        paths = filedialog.askopenfilenames(
            title="Selecione uma ou mais imagens",
            filetypes=[
                ("Imagens", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if not paths:
            return

        self.files = list(paths)
        self.listbox.delete(0, tk.END)
        for p in self.files:
            self.listbox.insert(tk.END, p)

        self.status.config(text=f"{len(self.files)} imagem(ns) selecionada(s).")

    def get_output_dir(self):
        if self.files:
            base = Path(self.files[0]).parent / "resultados_corrosao"
        else:
            base = Path.cwd() / "resultados_corrosao"
        base.mkdir(parents=True, exist_ok=True)
        return base

    def open_output(self):
        path = self.get_output_dir()
        try:
            os.startfile(path)  # Windows
        except Exception:
            messagebox.showinfo("Pasta", str(path))

    def show_image(self, path):
        img = Image.open(path).convert("RGB")
        img.thumbnail((760, 600))
        self.current_photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.current_photo, text="")

    def run_analysis(self):
        if not self.files:
            messagebox.showwarning("Atenção", "Selecione pelo menos uma imagem.")
            return

        try:
            px_per_mm = parse_px_per_mm(self.calibration_var.get())
        except ValueError as exc:
            messagebox.showwarning("Calibração inválida", str(exc))
            return

        output_dir = self.get_output_dir()
        paths = list(self.files)
        sensitivity = float(self.sens_var.get())

        self.select_button.config(state="disabled")
        self.analyze_button.config(state="disabled")
        self.progress.configure(maximum=len(paths), value=0)
        self.status.config(text="Iniciando análise...")
        self.analysis_thread = threading.Thread(
            target=self._analyze_files,
            args=(paths, output_dir, sensitivity, px_per_mm),
            daemon=True,
        )
        self.analysis_thread.start()
        self.after(100, self._poll_analysis_queue)

    def _analyze_files(self, paths, output_dir, sensitivity, px_per_mm):
        results = []
        errors = []
        last_overlay = None
        names = output_stems(paths)
        try:
            for index, (path, output_stem) in enumerate(zip(paths, names), start=1):
                self.analysis_queue.put(("progress", index, len(paths), path))
                try:
                    result, last_overlay = analyze(
                        path,
                        output_dir,
                        sens=sensitivity,
                        min_area=12,
                        px_per_mm=px_per_mm,
                        n_label=40,
                        output_stem=output_stem,
                    )
                    results.append(result)
                except Exception as exc:
                    errors.append((path, f"{type(exc).__name__}: {exc}"))

            csv_path = save_summary(results, output_dir)
            self.analysis_queue.put(
                ("finished", results, last_overlay, errors, csv_path, output_dir)
            )
        except Exception as exc:
            self.analysis_queue.put(("fatal", f"{type(exc).__name__}: {exc}"))

    def _poll_analysis_queue(self):
        try:
            while True:
                event = self.analysis_queue.get_nowait()
                if event[0] == "progress":
                    _, index, total, path = event
                    self.progress.configure(value=index - 1)
                    self.status.config(
                        text=f"Analisando {index}/{total}: {Path(path).name}"
                    )
                elif event[0] == "finished":
                    _, results, last_overlay, errors, csv_path, output_dir = event
                    self._finish_analysis(results, last_overlay, errors, csv_path, output_dir)
                    return
                elif event[0] == "fatal":
                    self._analysis_failed(event[1])
                    return
        except queue.Empty:
            pass
        self.after(100, self._poll_analysis_queue)

    def _finish_analysis(self, results, last_overlay, errors, csv_path, output_dir):
        self.progress.configure(value=len(self.files))
        self.select_button.config(state="normal")
        self.analyze_button.config(state="normal")
        self.analysis_thread = None

        if last_overlay is not None:
            self.show_image(last_overlay)

        lines = ["ANÁLISE CONCLUÍDA", "=" * 32, ""]
        for result in results:
            lines.append(f"Arquivo: {Path(result['arquivo']).name}")
            lines.append(f"Corrosão total: {result['pct_corrosao_total']:.2f}%")
            lines.append(f"Manchas: {result['n_manchas_total']}")
            lines.append(f"Maior mancha: {result['maior_mancha_px']} px")
            if "area_corrosao_total_mm2" in result:
                lines.append(f"Área corroída: {result['area_corrosao_total_mm2']:.3f} mm²")
                lines.append(f"Área da maior mancha: {result['maior_mancha_mm2']:.3f} mm²")
            lines.append("")

        if errors:
            lines.extend(["IMAGENS COM ERRO", ""])
            lines.extend(f"{Path(path).name}: {error}" for path, error in errors)
            self.status.config(
                text=f"Concluído com {len(errors)} falha(s); {len(results)} imagem(ns) processada(s)."
            )
        else:
            self.status.config(text="Concluído.")

        lines.append(f"CSV: {csv_path}" if csv_path else "Nenhuma imagem foi processada com sucesso.")
        lines.append(f"Pasta: {output_dir}")
        self.stats.delete("1.0", tk.END)
        self.stats.insert("1.0", "\n".join(lines))

        if errors:
            messagebox.showwarning(
                "Análise concluída com falhas",
                f"{len(results)} imagem(ns) processada(s); {len(errors)} falha(s).\n"
                f"Consulte os detalhes no painel e os resultados em:\n{output_dir}",
            )
        else:
            messagebox.showinfo(
                "Análise concluída",
                f"Processamento finalizado.\n\nResultados em:\n{output_dir}",
            )

    def _analysis_failed(self, error):
        self.select_button.config(state="normal")
        self.analyze_button.config(state="normal")
        self.analysis_thread = None
        self.status.config(text="Erro ao salvar os resultados.")
        messagebox.showerror("Erro na análise", error)




def main():
    App().mainloop()


if __name__ == "__main__":
    main()
