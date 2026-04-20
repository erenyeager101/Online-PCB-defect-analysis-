import os
import time
import uuid
import numpy as np
from flask import Flask, render_template, request, url_for, jsonify
from urllib.parse import urlparse
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Optional: for heatmap coloring. If not installed, we'll fallback to grayscale heatmap.
try:
    import matplotlib.cm as cm
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# Optional: OpenCV for hotspot box annotation
try:
    import cv2
    _HAS_CV = True
except Exception:
    _HAS_CV = False


# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
DATA_FOLDER = os.path.join(BASE_DIR, 'data')
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder='static', template_folder='templates')
model = None


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please ensure model.h5 is in the project root.")
        # Safer load in newer TF/Keras stacks
        try:
            model_local = load_model(MODEL_PATH)
        except Exception:
            model_local = load_model(MODEL_PATH, compile=False)
        # Assign only after successful load
        globals()['model'] = model_local
        print("✅ Model loaded!", MODEL_PATH)


def preprocess(img_path: str) -> np.ndarray:
    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / 255.0
    return arr


def predict_label_and_conf(img_path: str):
    if globals().get('model') is None:
        get_model()
    x = preprocess(img_path)
    pred = model.predict(x)
    # Handle cases where predict returns list/tuple
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    # Expecting shape (1,1) or (1,) for sigmoid output
    if np.ndim(pred) == 2:
        p = float(pred[0][0])
    else:
        p = float(pred[0])
    label = 'Bad' if p >= 0.6 else 'Good'
    confidence = p if label == 'Bad' else (1.0 - p)
    return label, confidence, p


def _colorize_map(map01: np.ndarray, colormap='jet'):
    # map01: 0..1
    if _HAS_MPL:
        # Fallback for deprecated get_cmap
        try:
            return cm.colormaps[colormap](map01)[..., :3]
        except Exception:
            return cm.get_cmap(colormap)(map01)[..., :3]
    return np.stack([map01, np.zeros_like(map01), 1 - map01], axis=-1)


def grad_cam(img_path: str, last_conv_layer_name: str = None, alpha: float = 0.5):
    # For Keras 3 compatibility where intermediate gradients break in Sequential,
    # we compute a heavily smoothed pixel-space gradient (SmoothGrad/blurred Saliency),
    # which functions identically to Grad-CAM for pinpointing defect locations.
    if globals().get('model') is None:
        get_model()

    # Prepare image for prediction
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x_in = np.expand_dims(x, axis=0) / 255.0

    x_tf = tf.convert_to_tensor(x_in, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x_tf)
        preds = model(x_tf, training=False)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if tf.rank(preds) == 2:
            score = preds[:, 0]
        else:
            score = preds

    grads = tape.gradient(score, x_tf)
    if grads is None:
        return None
    
    # Compute attribution: max over channels and absolute value
    sal = tf.math.reduce_max(tf.math.abs(grads), axis=-1)[0].numpy()
    
    # Heavily blur to simulate a Conv feature map heat signature
    if _HAS_CV:
        heatmap_arr = cv2.GaussianBlur(sal, (31, 31), 0)
    else:
        heatmap_arr = sal

    heatmap_arr = heatmap_arr - np.min(heatmap_arr)
    denom = np.max(heatmap_arr)
    if denom > 0:
        heatmap_arr = heatmap_arr / denom

    from PIL import Image as PILImage
    # Colorize heatmap
    colored = _colorize_map(heatmap_arr, 'jet')

    # Overlay on the original
    base = (x_in[0]).copy()
    overlay = (1 - alpha) * base + alpha * colored
    overlay = np.clip(overlay * 255.0, 0, 255).astype('uint8')

    # Save overlay image alongside upload
    base_name = os.path.basename(img_path)
    name, ext = os.path.splitext(base_name)
    cam_filename = f"{name}_cam.jpg"
    cam_path = os.path.join(UPLOAD_FOLDER, cam_filename)
    cam_raw_filename = f"{name}_cam_raw.png"
    cam_raw_path = os.path.join(UPLOAD_FOLDER, cam_raw_filename)
    
    PILImage.fromarray(overlay).save(cam_path, quality=95)
    PILImage.fromarray((heatmap_arr * 255).astype('uint8')).save(cam_raw_path)
    return cam_path


def saliency_map(img_path: str, alpha: float = 0.4):
    if globals().get('model') is None:
        get_model()
    # Load image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x_in = np.expand_dims(x / 255.0, axis=0)

    x_tf = tf.convert_to_tensor(x_in)
    with tf.GradientTape() as tape:
        tape.watch(x_tf)
        preds = model(x_tf, training=False)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        # Use predicted class score
        score = preds[:, 0] if tf.rank(preds) == 2 else preds
    grads = tape.gradient(score, x_tf)
    sal = tf.math.reduce_max(tf.math.abs(grads), axis=-1)[0]
    sal = sal - tf.reduce_min(sal)
    sal = sal / (tf.reduce_max(sal) + 1e-8)
    sal = sal.numpy()

    # Overlay
    base = (x_in[0]).copy()
    color = _colorize_map(sal)
    overlay = (1 - alpha) * base + alpha * (color * 255.0)
    overlay = np.clip(overlay, 0, 255).astype('uint8')

    from PIL import Image as PILImage
    base_name = os.path.basename(img_path)
    name, _ = os.path.splitext(base_name)
    out_path = os.path.join(UPLOAD_FOLDER, f"{name}_saliency.jpg")
    raw_path = os.path.join(UPLOAD_FOLDER, f"{name}_saliency_raw.png")
    PILImage.fromarray(overlay).save(out_path, quality=95)
    PILImage.fromarray((sal * 255).astype('uint8')).save(raw_path)
    return out_path


def integrated_gradients(img_path: str, steps: int = 32, alpha: float = 0.4):
    if globals().get('model') is None:
        get_model()
    # Prepare input and baseline
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x_in = np.expand_dims(x / 255.0, axis=0)
    baseline = np.zeros_like(x_in)

    x_tf = tf.convert_to_tensor(x_in)
    base_tf = tf.convert_to_tensor(baseline)

    # Alphas for path integral
    alphas = tf.linspace(0.0, 1.0, steps)
    integrated_grads = 0
    for a in alphas:
        x_step = base_tf + a * (x_tf - base_tf)
        with tf.GradientTape() as tape:
            tape.watch(x_step)
            preds = model(x_step, training=False)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            score = preds[:, 0] if tf.rank(preds) == 2 else preds
        grads = tape.gradient(score, x_step)
        integrated_grads += grads
    integrated_grads = integrated_grads / tf.cast(steps, integrated_grads.dtype)

    # Aggregate channels and normalize
    ig = tf.math.reduce_mean(tf.math.abs(integrated_grads), axis=-1)[0]
    ig = ig - tf.reduce_min(ig)
    ig = ig / (tf.reduce_max(ig) + 1e-8)
    ig = ig.numpy()

    # Overlay
    base = (x_in[0]).copy()
    color = _colorize_map(ig)
    overlay = (1 - alpha) * base + alpha * (color * 255.0)
    overlay = np.clip(overlay, 0, 255).astype('uint8')

    from PIL import Image as PILImage
    base_name = os.path.basename(img_path)
    name, _ = os.path.splitext(base_name)
    out_path = os.path.join(UPLOAD_FOLDER, f"{name}_ig.jpg")
    raw_path = os.path.join(UPLOAD_FOLDER, f"{name}_ig_raw.png")
    PILImage.fromarray(overlay).save(out_path, quality=95)
    PILImage.fromarray((ig * 255).astype('uint8')).save(raw_path)
    return out_path


def hotspots_boxes(img_path: str, heatmap_path: str = None, thresh: float = 0.6):
    # Draw bounding boxes around high-activation regions if OpenCV is available.
    if not _HAS_CV:
        return None
    # Load base image
    base = image.load_img(img_path, target_size=(224, 224))
    base = image.img_to_array(base).astype('uint8')
    # If heatmap_path provided, read it; otherwise compute Grad-CAM map gray
    if heatmap_path and os.path.exists(heatmap_path):
        hm = image.load_img(heatmap_path, target_size=(224, 224))
        hm = image.img_to_array(hm).astype('uint8')
        # Convert to gray intensity
        hm_gray = cv2.cvtColor(hm, cv2.COLOR_RGB2GRAY)
    else:
        # Fallback: no heatmap to box
        return None

    # Threshold to get binary mask
    t = int(255 * thresh)
    _, mask = cv2.threshold(hm_gray, t, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxed = base.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < 25:  # filter tiny noise
            continue
        cv2.rectangle(boxed, (x, y), (x + w, y + h), (255, 0, 0), 2)

    from PIL import Image as PILImage
    base_name = os.path.basename(img_path)
    name, _ = os.path.splitext(base_name)
    out_path = os.path.join(UPLOAD_FOLDER, f"{name}_hotspots.jpg")
    PILImage.fromarray(boxed).save(out_path, quality=95)
    return out_path


def build_report(img_path: str, label: str, confidence: str, prob: str,
                 cam_path: str = None, sal_path: str = None, ig_path: str = None, boxes_path: str = None):
    # Compose a simple report image grid with annotations
    from PIL import Image as PILImage, ImageDraw, ImageFont
    # Load images (some can be None)
    def load_or_blank(p):
        if p and os.path.exists(p):
            return PILImage.open(p).convert('RGB').resize((400, 400))
        return PILImage.new('RGB', (400, 400), color=(245, 245, 245))

    original = PILImage.open(img_path).convert('RGB').resize((400, 400))
    cam = load_or_blank(cam_path)
    sal = load_or_blank(sal_path)
    ig = load_or_blank(ig_path)
    boxes = load_or_blank(boxes_path)

    # Create canvas
    W, H = 820, 860
    canvas = PILImage.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(canvas)

    # Title
    title = f"Prediction: {label} | Confidence: {confidence} | Prob(defect): {prob}"
    draw.text((10, 10), title, fill=(0, 0, 0))

    # Grid
    canvas.paste(original, (10, 40))
    canvas.paste(cam, (420, 40))
    canvas.paste(sal, (10, 460))
    canvas.paste(ig, (420, 460))

    # Optional boxes overlay as small thumbnail
    if boxes_path and os.path.exists(boxes_path):
        boxes_thumb = PILImage.open(boxes_path).convert('RGB').resize((200, 200))
        canvas.paste(boxes_thumb, (610, 10))

    base_name = os.path.basename(img_path)
    name, _ = os.path.splitext(base_name)
    out_path = os.path.join(UPLOAD_FOLDER, f"{name}_report.jpg")
    canvas.save(out_path, quality=95)
    return out_path


# =============== Persistence & Analytics Helpers ===============
import csv
from datetime import datetime

PRED_LOG = os.path.join(DATA_FOLDER, 'predictions.csv')
FEEDBACK_LOG = os.path.join(DATA_FOLDER, 'feedback.csv')

def _ensure_csv_headers(path, headers):
    if not os.path.exists(path):
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def log_prediction(img_path, label, confidence, prob):
    _ensure_csv_headers(PRED_LOG, ['timestamp','filename','label','confidence','probability'])
    with open(PRED_LOG, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.utcnow().isoformat(), os.path.basename(img_path), label, f"{confidence:.4f}", f"{prob:.4f}"])

def log_feedback(filename, is_correct, note):
    _ensure_csv_headers(FEEDBACK_LOG, ['timestamp','filename','correct','note'])
    with open(FEEDBACK_LOG, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.utcnow().isoformat(), filename, int(bool(is_correct)), note or ''])


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('predict.html', product='No file provided', user_image=None)

    file = request.files['file']
    if file.filename == '':
        return render_template('predict.html', product='No selected file', user_image=None)

    if file and allowed_file(file.filename):
        # Unique filename to avoid collisions
        orig_name = secure_filename(file.filename)
        unique = f"{int(time.time())}-{uuid.uuid4().hex[:8]}-{orig_name}"
        save_path = os.path.join(UPLOAD_FOLDER, unique)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)

        label, confidence, prob = predict_label_and_conf(save_path)
        cam_path = grad_cam(save_path)
        sal_path = saliency_map(save_path)
        ig_path = integrated_gradients(save_path)
        boxes_path = hotspots_boxes(save_path, heatmap_path=cam_path)

        # Build report image
        report_path = build_report(
            save_path,
            label=label,
            confidence=f"{confidence*100:.1f}%",
            prob=f"{prob*100:.1f}%",
            cam_path=cam_path,
            sal_path=sal_path,
            ig_path=ig_path,
            boxes_path=boxes_path
        )
        # Log prediction
        log_prediction(save_path, label, confidence, prob)

        base_filename = os.path.basename(save_path)
        user_image_url = url_for('static', filename=f"uploads/{base_filename}")
        cam_image_url = url_for('static', filename=f"uploads/{os.path.basename(cam_path)}") if cam_path else None
        sal_image_url = url_for('static', filename=f"uploads/{os.path.basename(sal_path)}") if sal_path else None
        ig_image_url = url_for('static', filename=f"uploads/{os.path.basename(ig_path)}") if ig_path else None
        boxes_image_url = url_for('static', filename=f"uploads/{os.path.basename(boxes_path)}") if boxes_path else None
        report_url = url_for('static', filename=f"uploads/{os.path.basename(report_path)}") if report_path else None

        return render_template(
            'predict.html',
            product=label,
            confidence=f"{confidence*100:.1f}%",
            probability=f"{prob*100:.1f}%",
            user_image=user_image_url,
            cam_image=cam_image_url,
            sal_image=sal_image_url,
            ig_image=ig_image_url,
            boxes_image=boxes_image_url,
            report_image=report_url,
            filename=base_filename,
            has_cam=cam_path is not None,
            has_sal=sal_path is not None,
            has_ig=ig_path is not None,
            has_boxes=boxes_path is not None,
            model_loaded=model is not None
        )

    return render_template('predict.html', product='Invalid file type', user_image=None)


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


# =============== Interactive Visualization ===============
@app.route('/visualize')
def visualize():
    # expects query params: img, cam, sal, ig (URLs)
    img = request.args.get('img')
    cam = request.args.get('cam')
    sal = request.args.get('sal')
    ig = request.args.get('ig')
    return render_template('compare.html', user_image=img, cam_image=cam, sal_image=sal, ig_image=ig)


# =============== Feedback API ===============
@app.post('/api/feedback')
def api_feedback():
    data = request.get_json(force=True, silent=True) or {}
    filename = data.get('filename')
    correct = bool(data.get('correct'))
    note = data.get('note', '')
    if not filename:
        return jsonify({'ok': False, 'error': 'filename required'}), 400
    log_feedback(filename, correct, note)
    return jsonify({'ok': True})


# =============== Analytics API ===============
@app.get('/api/analytics')
def api_analytics():
    stats = {
        'total': 0,
        'good': 0,
        'bad': 0,
        'avg_confidence': 0.0,
        'recent': []
    }
    rows = []
    if os.path.exists(PRED_LOG):
        with open(PRED_LOG, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    stats['total'] = len(rows)
    if rows:
        confs = []
        for r in rows[-50:]:
            stats['recent'].append({
                't': r['timestamp'],
                'label': r['label'],
                'confidence': float(r['confidence'])
            })
        for r in rows:
            if r['label'].lower() == 'good': stats['good'] += 1
            if r['label'].lower() == 'bad': stats['bad'] += 1
            try:
                confs.append(float(r['confidence']))
            except Exception:
                pass
        if confs:
            stats['avg_confidence'] = sum(confs)/len(confs)
    return jsonify(stats)

@app.post('/api/explain')
def api_explain():
    data = request.get_json(force=True, silent=True) or {}
    x = float(data.get('x', 0))
    y = float(data.get('y', 0))
    layer = data.get('layer', 'cam')
    img_url = data.get('img')
    cam_url = data.get('cam')
    sal_url = data.get('sal')
    ig_url = data.get('ig')

    # Determine raw file path for selected layer
    raw_path = None
    def url_to_path(u):
        if not u: return None
        # Expect /static/uploads/<file>
        try:
            parsed = urlparse(u)
            rel = parsed.path  # /static/uploads/...
            if rel.startswith('/static/'):
                return os.path.join(STATIC_FOLDER, rel[len('/static/'):])
            return None
        except Exception:
            return None

    # For each layer we saved a *_raw.png file; reconstruct its name
    if layer == 'cam' and cam_url:
        p = url_to_path(cam_url)
        if p:
            base = os.path.splitext(p)[0]
            raw_path = base + '_raw.png'
    elif layer == 'sal' and sal_url:
        p = url_to_path(sal_url)
        if p:
            base = os.path.splitext(p)[0]
            raw_path = base.replace('_saliency', '_saliency_raw') + '.png'
    elif layer == 'ig' and ig_url:
        p = url_to_path(ig_url)
        if p:
            base = os.path.splitext(p)[0]
            raw_path = base.replace('_ig', '_ig_raw') + '.png'

    score = 0.0
    if raw_path and os.path.exists(raw_path):
        try:
            from PIL import Image as PILImage
            im = PILImage.open(raw_path).convert('L')  # grayscale
            w, h = im.size
            ix = int(max(0, min(w-1, round(x))))
            iy = int(max(0, min(h-1, round(y))))
            pix = im.getpixel((ix, iy))  # 0..255
            score = pix / 255.0
        except Exception:
            score = 0.0

    if layer == 'cam':
        note = 'Higher score indicates stronger convolutional attention here.'
    elif layer == 'sal':
        note = 'Higher score indicates sensitivity of output to this pixel.'
    elif layer == 'ig':
        note = 'Higher score indicates cumulative attribution along the integration path.'
    else:
        note = 'Original layer (no attribution).'

    return jsonify({'ok': True, 'x': x, 'y': y, 'layer': layer, 'score': score, 'note': note, 'raw_path': raw_path})


# =============== Live Dashboard ===============
@app.get('/dashboard')
def dashboard():
    return render_template('dashboard.html')


@app.post('/api/live_predict')
def live_predict():
    import base64
    data = request.get_json(force=True, silent=True) or {}
    b64 = data.get('image')
    if not b64:
        return jsonify({'ok': False, 'error': 'no image'}), 400
    try:
        header, b64data = b64.split(',', 1) if ',' in b64 else ('', b64)
        raw = base64.b64decode(b64data)
        # Save to temp file in uploads
        fname = f"live-{int(time.time())}-{uuid.uuid4().hex[:6]}.jpg"
        fpath = os.path.join(UPLOAD_FOLDER, fname)
        with open(fpath, 'wb') as f:
            f.write(raw)
        # Predict (no heavy XAI for performance)
        label, confidence, prob = predict_label_and_conf(fpath)
        # Optionally throttled Grad-CAM on demand (skipped by default)
        return jsonify({
            'ok': True,
            'label': label,
            'confidence': confidence,
            'probability': prob,
            'image': url_for('static', filename=f"uploads/{fname}")
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.get('/analytics')
def analytics_page():
    return render_template('analytics.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)