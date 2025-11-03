import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXT = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# --- Image processing functions ---
def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def filtering_mean(img_gray, k=5):
    return cv2.blur(img_gray, (k, k))

def filtering_median(img_gray, k=5):
    k = max(3, k if k % 2 == 1 else k+1)
    return cv2.medianBlur(img_gray, k)

def filtering_gaussian(img_gray, k=5):
    k = max(3, k if k % 2 == 1 else k+1)
    return cv2.GaussianBlur(img_gray, (k, k), 0)

def edge_density(img_gray):
    edges = cv2.Canny(img_gray, 100, 200)
    edge_pixels = np.count_nonzero(edges)
    total = edges.size
    density = edge_pixels / total if total > 0 else 0
    return density, edges

# --- Heuristic probability ---
def compute_heu_prob(mean_intensity, edge_density_val, mean_blue):
    mean_norm = mean_intensity / 255.0
    blue_norm = mean_blue / 255.0
    edge = edge_density_val

    score_cerah = 0.3 * mean_norm + 0.5 * blue_norm + 0.2 * (1 - edge)
    score_mendung = 0.4 * (1 - mean_norm) + 0.4 * (1 - blue_norm) + 0.2 * edge
    target_mean = 0.6
    dist_from_target_mean = abs(mean_norm - target_mean) * 2
    score_berawan = 0.6 * (1 - dist_from_target_mean) + 0.2 * (1 - blue_norm) + 0.2 * edge

    scores = np.array([score_cerah, score_berawan, score_mendung])
    scores = np.clip(scores, 0, None)
    ssum = scores.sum()
    probs = scores / ssum if ssum != 0 else np.array([1/3, 1/3, 1/3])
    probs_int = (probs * 100).round().astype(int)
    if probs_int.sum() != 100:
        probs_int[np.argmax(probs_int)] += 100 - probs_int.sum()
    return probs_int.tolist()

# --- Simulate compression ---
def simulate_compression(original_path):
    sizes = {}
    base = os.path.basename(original_path)
    name, _ = os.path.splitext(base)
    img = Image.open(original_path).convert('RGB')

    out_jpeg_85 = os.path.join(app.config['UPLOAD_FOLDER'], f"{name}_q85.jpg")
    out_jpeg_40 = os.path.join(app.config['UPLOAD_FOLDER'], f"{name}_q40.jpg")
    out_png = os.path.join(app.config['UPLOAD_FOLDER'], f"{name}_opt.png")

    img.save(out_jpeg_85, format='JPEG', quality=85, optimize=True)
    img.save(out_jpeg_40, format='JPEG', quality=40, optimize=True)
    img.save(out_png, format='PNG', optimize=True)

    sizes['orig'] = os.path.getsize(original_path)
    sizes['jpg_q85'] = os.path.getsize(out_jpeg_85)
    sizes['jpg_q40'] = os.path.getsize(out_jpeg_40)
    sizes['png_opt'] = os.path.getsize(out_png)

    def hr(n):
        for u in ['B','KB','MB']:
            if n < 1024:
                return f"{n:.0f} {u}"
            n /= 1024
        return f"{n:.2f} GB"

    sizes_hr = {k: hr(v) for k,v in sizes.items()}
    return sizes_hr, {
        'jpg_q85': os.path.basename(out_jpeg_85),
        'jpg_q40': os.path.basename(out_jpeg_40),
        'png_opt': os.path.basename(out_png)
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or not allowed_file(file.filename):
            return redirect(request.url)
        filename = file.filename
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        img_bgr = cv2.imdecode(np.fromfile(save_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            pil = Image.open(save_path).convert('RGB')
            arr = np.array(pil)[:, :, ::-1].copy()
            img_bgr = arr

        gray = to_grayscale(img_bgr)
        mean_filtered = filtering_mean(gray)
        median_filtered = filtering_median(gray)
        gaussian_filtered = filtering_gaussian(gray)
        ed_density, edges = edge_density(gray)

        height, width = img_bgr.shape[:2]
        crop_height = int(height * 0.60)
        sky_area_bgr = img_bgr[:crop_height, :]
        sky_area_gray = gray[:crop_height, :]
        mean_int = float(np.mean(sky_area_gray))
        mean_blue = float(np.mean(sky_area_bgr[:, :, 0]))

        sky_mask = np.zeros(gray.shape, dtype=np.uint8)
        sky_mask[:crop_height, :] = 255

        probs = compute_heu_prob(mean_int, ed_density, mean_blue)
        sizes_hr, comp_files = simulate_compression(save_path)

        def save_stage(name, data):
            path = os.path.join(app.config['UPLOAD_FOLDER'], f"{name}_{filename}")
            cv2.imwrite(path, data)
            return os.path.basename(path)

        result = {
            'filename': filename,
            'mean_intensity': round(mean_int, 2),
            'mean_blue': round(mean_blue, 2),
            'edge_density': round(ed_density, 4),
            'probs': {
                'cerah': int(probs[0]),
                'berawan': int(probs[1]),
                'mendung': int(probs[2])
            },
            'sizes_hr': sizes_hr,
            'comp_files': comp_files,
            'process': {
                'gray': save_stage("gray", gray),
                'mean': save_stage("mean", mean_filtered),
                'median': save_stage("median", median_filtered),
                'gauss': save_stage("gauss", gaussian_filtered),
                'edge': save_stage("edge", edges),
                'mask': save_stage("mask", sky_mask)
            }
        }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
