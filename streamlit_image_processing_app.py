"""
ImageProcessingLectures_app.py
Interactive Image Processing Lectures using Streamlit + OpenCV + NumPy + Pillow

Run:
    pip install streamlit opencv-python-headless numpy pillow
    streamlit run ImageProcessingLectures_app.py

Structure: single-file app with 9 lecture modules implemented as functions.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import os
from typing import Tuple

st.set_page_config(page_title="Image Processing Lectures", layout="wide")

# --------------------- Utilities ---------------------
@st.cache_resource
def load_sample_image(path: str = None) -> np.ndarray:
    """Load a sample image from disk. If path is None, generate a synthetic image."""
    if path and os.path.exists(path):
        img = Image.open(path).convert("RGB")
        return np.array(img)
    # synthetic sample: gradient + shapes
    h, w = 512, 768
    base = np.zeros((h, w, 3), dtype=np.uint8)
    # gradient
    for i in range(h):
        color = int(255 * (i / h))
        base[i, :, :] = (color // 2, color, 255 - color)
    # add a rectangle and circle
    cv2.rectangle(base, (50, 50), (250, 200), (0, 255, 0), -1)
    cv2.circle(base, (500, 300), 80, (255, 0, 0), -1)
    return base


def to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) if img.ndim == 3 else Image.fromarray(img)


def read_image(uploaded_file) -> np.ndarray:
    image = Image.open(uploaded_file).convert('RGB')
    return np.array(image)


def show_before_after(img_before: np.ndarray, img_after: np.ndarray, caption_before: str = "Before", caption_after: str = "After"):
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_before, caption=caption_before, use_column_width=True)
    with col2:
        st.image(img_after, caption=caption_after, use_column_width=True)


def save_image_to_bytes(img: np.ndarray, fmt: str = 'PNG') -> bytes:
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) if img.ndim == 3 else Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return buf.getvalue()


def ensure_bgr(img: np.ndarray) -> np.ndarray:
    # convert RGB (PIL) -> BGR (OpenCV style) if needed
    if img.ndim == 3 and img.shape[2] == 3:
        # PIL->numpy gives RGB, convert to BGR for OpenCV ops convenience
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

# --------------------- App Layout ---------------------
st.title("📚 سلسلة محاضرات تفاعلية في معالجة الصور")
st.sidebar.header("المحاضرات")
lectures = [
    "محاضرة 1: مدخل ومعمارية الصور الرقمية",
    "محاضرة 2: أنظمة الألوان",
    "محاضرة 3: العمليات على البكسل",
    "محاضرة 4: الفلاتر والالتفاف",
    "محاضرة 5: إزالة الضوضاء",
    "محاضرة 6: كشف الحواف",
    "محاضرة 7: العمليات المورفولوجية",
    "محاضرة 8: التحويلات الهندسية",
    "محاضرة 9: المشروع الختامي"
]
choice = st.sidebar.selectbox("اختر المحاضرة", lectures)

# Upload image / or use sample
st.sidebar.markdown("---")
use_sample = st.sidebar.checkbox("استخدم صورة افتراضية (عند عدم رفع صورة)", value=True)
uploaded = st.sidebar.file_uploader("ارفع صورة (jpg, png)", type=["jpg", "jpeg", "png"])
if uploaded:
    user_img = read_image(uploaded)
else:
    user_img = load_sample_image()

# Convert to BGR for OpenCV processing; keep a copy of original RGB for display consistency
orig_rgb = user_img.copy()
orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR) if orig_rgb.ndim == 3 else orig_rgb.copy()

# Utility: display image info
def image_info(img: np.ndarray):
    if img is None:
        return
    st.write(f"**الأبعاد (Height × Width × Channels):** {img.shape}")
    st.write(f"**نوع البيانات:** {img.dtype}")
    # estimate bit depth from dtype
    bit_depth = {
        'uint8': 8,
        'uint16': 16,
        'float32': 32
    }.get(str(img.dtype), 'unknown')
    st.write(f"**العمق اللوني (تقريبي):** {bit_depth} bits")

# --------------------- Lecture Implementations ---------------------

# Lecture 1
def lecture1(img_rgb: np.ndarray):
    st.header("📘 المحاضرة 1: مدخل ومعمارية الصور الرقمية")
    st.markdown(
        """
- الصورة الرقمية عبارة عن مصفوفة من البكسلات، وكل بكسل يُمثل قيمة أو مجموعة قيم تحدد اللون.
- الأبعاد (Height × Width × Channels) تحدد دقة الصورة وشكلها.
- العمق اللوني (bit depth) يحدد عدد القيم الممكنة لكل قناة (مثلاً 8-bit → 0–255).
- صيغ الملفات (JPEG, PNG) تؤثر على الضغط والجودة.
- فهم تمثيل الصورة مهم قبل تطبيق أي معالجة.
        """
    )

    st.subheader("التطبيق")
    st.write("رفع صورة / استخدام صورة افتراضية → عرض معلومات الصورة → عرض الصورة الأصلية")
    image_info(img_rgb)
    st.image(img_rgb, caption="الصورة الأصلية", use_column_width=True)

# Lecture 2
def lecture2(img_rgb: np.ndarray):
    st.header("🎨 المحاضرة 2: أنظمة الألوان (Color Spaces)")
    st.markdown(
        """
- أشهر أنظمة الألوان: RGB (لعرض الشاشات)، Gray (أحادي)، HSV (مفيد للتعامل مع اللون/تشبع/سطوع).
- التحويل بين الأنظمة مفيد لتبسيط مهام مثل الكشف عن ألوان محددة أو المعالجة الهندسية.
- بعض المكتبات (OpenCV) تستخدم BGR افتراضياً.
        """
    )

    st.subheader("التطبيق")
    mode = st.selectbox("اختر التحويل", ["RGB (أصلية)", "Gray", "HSV", "فصل قنوات R/G/B"])
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if mode == "RGB (أصلية)":
        out = img_rgb
    elif mode == "Gray":
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        out = gray
    elif mode == "HSV":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        # convert HSV to RGB for display (scale okay)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    else:
        b, g, r = cv2.split(bgr)
        # stack as images for display
        r_img = cv2.merge([r, r, r])
        g_img = cv2.merge([g, g, g])
        b_img = cv2.merge([b, b, b])
        st.write("### قنوات R / G / B")
        cols = st.columns(3)
        cols[0].image(cv2.cvtColor(r_img, cv2.COLOR_BGR2RGB), caption="R", use_column_width=True)
        cols[1].image(cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB), caption="G", use_column_width=True)
        cols[2].image(cv2.cvtColor(b_img, cv2.COLOR_BGR2RGB), caption="B", use_column_width=True)
        return

    show_before_after(img_rgb, out, "الأصلية", "النتيجة")

# Lecture 3
def lecture3(img_rgb: np.ndarray):
    st.header("⚙️ المحاضرة 3: العمليات على البكسل (Point Operations)")
    st.markdown(
        """
- تعديل السطوع: إضافة/طرح قيمة من كل بكسل.
- تعديل التباين: ضرب/تغيير مدى القيم.
- الصور السالبة: 255 - قيمة البكسل.
- Thresholding: تحويل إلى ثنائية عن طريق عتبة.
        """
    )

    st.subheader("التطبيق")
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    brightness = st.slider("السطوع (مضاف)", -100, 100, 0)
    contrast = st.slider("التباين (مضاعف)", 50, 200, 100)
    apply_negative = st.button("تطبيق Negative")

    # apply brightness & contrast
    alpha = contrast / 100.0
    beta = brightness
    adjusted = cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)

    if apply_negative:
        neg = 255 - adjusted
        result = neg
    else:
        result = adjusted

    st.write("### Thresholding")
    thresh_type = st.selectbox("نوع Threshold", ["Binary", "Otsu"])
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    if thresh_type == "Binary":
        t = st.slider("العتبة", 0, 255, 127)
        _, binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    show_before_after(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(result, cv2.COLOR_BGR2RGB), "الأصلية", "بعد السطوع/التباين/السلبي")
    st.image(binary, caption="الـ Threshold (ثنائية)", use_column_width=True)

# Lecture 4

def lecture4(img_rgb: np.ndarray):
    st.header("🧪 المحاضرة 4: الفلاتر والالتفاف (Filtering & Convolution)")
    st.markdown(
        """
- الفلاتر تعتمد على Kernel/Mask تُطبق عبر الالتفاف (Convolution).
- أمثلة: Blur (تنعيم)، Sharpen (حماية الحواف)، Emboss (تأثير نحت).
- يمكن التحكم بحجم Kernel في Gaussian/Median.
        """
    )

    st.subheader("التطبيق")
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    choice = st.selectbox("اختر فلترًا", ["Original", "Gaussian Blur", "Median Blur", "Sharpen", "Emboss", "Edge (Laplacian)"])

    if choice == "Original":
        out = bgr
    elif choice == "Gaussian Blur":
        k = st.slider("Kernel size (odd)", 1, 31, 5, step=2)
        out = cv2.GaussianBlur(bgr, (k, k), 0)
    elif choice == "Median Blur":
        k = st.slider("Kernel size (odd)", 1, 31, 5, step=2)
        out = cv2.medianBlur(bgr, k)
    elif choice == "Sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        out = cv2.filter2D(bgr, -1, kernel)
    elif choice == "Emboss":
        kernel = np.array([[ -2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        out = cv2.filter2D(bgr, -1, kernel) + 128
    else:
        lap = cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
        lap = np.uint8(np.absolute(lap))
        out = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)

    show_before_after(img_rgb, cv2.cvtColor(out, cv2.COLOR_BGR2RGB), "الأصلية", choice)

# Lecture 5

def add_salt_pepper(img: np.ndarray, amount=0.004, s_vs_p=0.5):
    out = img.copy()
    h, w = out.shape[:2]
    num_salt = np.ceil(amount * h * w * s_vs_p)
    num_pepper = np.ceil(amount * h * w * (1.0 - s_vs_p))

    # Salt
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in out.shape[:2]]
    out[coords[0], coords[1]] = 255

    # Pepper
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in out.shape[:2]]
    out[coords[0], coords[1]] = 0
    return out


def add_gaussian_noise(img: np.ndarray, mean=0, var=10):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, img.shape).reshape(img.shape)
    noisy = img + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def lecture5(img_rgb: np.ndarray):
    st.header("🔊 المحاضرة 5: إزالة الضوضاء (Denoising)")
    st.markdown(
        """
- أنواع الضوضاء: Salt & Pepper، Gaussian.
- فلتر Median جيد لإزالة الضوضاء الملح والفلفل.
- Bilateral filter يحافظ على الحواف أثناء التنعيم.
        """
    )

    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    add_noise = st.checkbox("أضف ضوضاء صناعية للصورة؟")
    noisy = bgr.copy()
    if add_noise:
        noise_type = st.selectbox("نوع الضوضاء", ["Salt & Pepper", "Gaussian"])
        if noise_type == "Salt & Pepper":
            amount = st.slider("مقدار الضوضاء (amount)", 0.001, 0.05, 0.004, step=0.001)
            noisy = add_salt_pepper(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), amount=amount)
            noisy = cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR)
        else:
            var = st.slider("التباين (var)", 1, 100, 10)
            noisy = add_gaussian_noise(bgr, var=var)

    method = st.selectbox("اختر طريقة التنقية", ["Median", "Bilateral", "Gaussian Blur"])
    if method == "Median":
        k = st.slider("Kernel (odd)", 1, 31, 5, step=2)
        denoised = cv2.medianBlur(noisy, k)
    elif method == "Bilateral":
        d = st.slider("d (pixel diameter)", 1, 50, 9)
        sigmaColor = st.slider("sigmaColor", 1, 200, 75)
        sigmaSpace = st.slider("sigmaSpace", 1, 200, 75)
        denoised = cv2.bilateralFilter(noisy, d, sigmaColor, sigmaSpace)
    else:
        k = st.slider("Kernel size (odd)", 1, 31, 5, step=2)
        denoised = cv2.GaussianBlur(noisy, (k, k), 0)

    show_before_after(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB), "الأصلية", "بعد التنقية")

# Lecture 6

def lecture6(img_rgb: np.ndarray):
    st.header("🧭 المحاضرة 6: كشف الحواف (Edge Detection)")
    st.markdown(
        """
- الحواف تمثل تغيّرات سريعة في شدة الصورة.
- طرق شائعة: Sobel (تدرجات x/y)، Laplacian، وCanny (طرق متعددة الخطوات).
- Canny يعتمد على عتبات منخفضة وعالية.
        """
    )

    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    method = st.selectbox("اختر طريقة كشف الحواف", ["Sobel", "Laplacian", "Canny"])

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if method == "Sobel":
        k = st.slider("Kernel size (odd)", 1, 31, 3, step=2)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=k)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=k)
        mag = np.sqrt(sx ** 2 + sy ** 2)
        mag = np.uint8(np.clip(mag / mag.max() * 255, 0, 255))
        out = mag
    elif method == "Laplacian":
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        out = np.uint8(np.absolute(lap))
    else:
        low = st.slider("العتبة المنخفضة (low)", 0, 255, 50)
        high = st.slider("العتبة العالية (high)", 0, 255, 150)
        out = cv2.Canny(gray, low, high)

    show_before_after(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), out if out.ndim == 2 else cv2.cvtColor(out, cv2.COLOR_BGR2RGB), "الأصلية", f"Edges: {method}")

# Lecture 7

def lecture7(img_rgb: np.ndarray):
    st.header("🔬 المحاضرة 7: العمليات المورفولوجية (Morphological Ops)")
    st.markdown(
        """
- عمليات مورفولوجية تعمل على الصور الثنائية (Binary) مثل Erosion وDilation.
- تُستخدم لإزالة الضوضاء الصغيرة، تعبئة الفجوات، فصل الكائنات، إلخ.
        """
    )

    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    t = st.slider("عتبة التحويل إلى ثنائية", 0, 255, 127)
    _, binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

    op = st.selectbox("اختر العملية المورفولوجية", ["Erosion", "Dilation", "Opening", "Closing"])
    ksize = st.slider("حجم العنصر البنائي (odd)", 1, 31, 3, step=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

    if op == "Erosion":
        result = cv2.erode(binary, kernel, iterations=1)
    elif op == "Dilation":
        result = cv2.dilate(binary, kernel, iterations=1)
    elif op == "Opening":
        result = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    else:
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    show_before_after(binary, result, "Binary", op)

# Lecture 8

def lecture8(img_rgb: np.ndarray):
    st.header("🔁 المحاضرة 8: التحويلات الهندسية (Geometric Transforms)")
    st.markdown(
        """
- تحويلات هندسية تشمل الترجمة، الدوران، التكبير/التصغير، الانعكاس، والقص.
- هذه العمليات تغير إحداثيات البكسل باستخدام مصفوفات تحويل.
        """
    )

    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    op = st.selectbox("اختر التحويل", ["Rotate", "Scale (Zoom)", "Translate", "Flip", "Crop"])

    h, w = bgr.shape[:2]
    if op == "Rotate":
        angle = st.slider("الزاوية (درجة)", -180, 180, 0)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        out = cv2.warpAffine(bgr, M, (w, h))
    elif op == "Scale (Zoom)":
        fx = st.slider("تكبير X", 0.1, 3.0, 1.0)
        fy = st.slider("تكبير Y", 0.1, 3.0, 1.0)
        out = cv2.resize(bgr, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    elif op == "Translate":
        tx = st.slider("تحريك X (px)", -w, w, 0)
        ty = st.slider("تحريك Y (px)", -h, h, 0)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        out = cv2.warpAffine(bgr, M, (w, h))
    elif op == "Flip":
        mode = st.selectbox("نوع الانعكاس", ["عمودي", "أفقي", "كلي"])
        if mode == "عمودي":
            out = cv2.flip(bgr, 0)
        elif mode == "أفقي":
            out = cv2.flip(bgr, 1)
        else:
            out = cv2.flip(bgr, -1)
    else:  # Crop
        st.write("اسحب مستطيل القص (أدخل إحداثيات)")
        x1 = st.number_input("x1", min_value=0, max_value=w - 1, value=0)
        y1 = st.number_input("y1", min_value=0, max_value=h - 1, value=0)
        x2 = st.number_input("x2", min_value=1, max_value=w, value=w)
        y2 = st.number_input("y2", min_value=1, max_value=h, value=h)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        out = bgr[y1:y2, x1:x2]

    # show
    show_before_after(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(out, cv2.COLOR_BGR2RGB) if out.ndim == 3 else out, "الأصلية", op)

# Lecture 9 - Pipeline

def apply_pipeline(bgr: np.ndarray, pipeline: list) -> np.ndarray:
    img = bgr.copy()
    for op in pipeline:
        name = op[0]
        params = op[1] if len(op) > 1 else {}
        if name == 'Gray':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif name == 'Blur':
            k = params.get('k', 5)
            img = cv2.GaussianBlur(img, (k, k), 0)
        elif name == 'Median':
            k = params.get('k', 5)
            img = cv2.medianBlur(img, k)
        elif name == 'Edges':
            low = params.get('low', 50)
            high = params.get('high', 150)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, low, high)
            img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif name == 'Thresh':
            t = params.get('t', 127)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)
            img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        elif name == 'Sharpen':
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            img = cv2.filter2D(img, -1, kernel)
        # add more ops as needed
    return img


def lecture9(img_rgb: np.ndarray):
    st.header("🏁 المحاضرة 9: المشروع الختامي — بناء Pipeline")
    st.markdown(
        """
- ارفع صورة أو استخدم الافتراضية.
- اختر سلسلة عمليات (مثلاً: Gray → Blur → Edges).
- عرض النتيجة النهائية وإمكانية تنزيل الصورة.
        """
    )

    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    ops = st.multiselect("اختر العمليات بالترتيب (اضغط وحرّك لإعادة الترتيب)", ['Gray', 'Blur', 'Median', 'Sharpen', 'Thresh', 'Edges'])
    # For simplicity, collect a single parameter for all where needed
    pipeline = []
    for op in ops:
        if op == 'Blur':
            k = st.slider('Blur kernel (for Blur)', 1, 31, 5, step=2)
            pipeline.append((op, {'k': k}))
        elif op == 'Median':
            k = st.slider('Median kernel (for Median)', 1, 31, 5, step=2)
            pipeline.append((op, {'k': k}))
        elif op == 'Thresh':
            t = st.slider('Threshold value (for Thresh)', 0, 255, 127)
            pipeline.append((op, {'t': t}))
        elif op == 'Edges':
            low = st.slider('Canny low threshold', 0, 255, 50)
            high = st.slider('Canny high threshold', 0, 255, 150)
            pipeline.append((op, {'low': low, 'high': high}))
        else:
            pipeline.append((op, {}))

    if st.button('تطبيق الـ Pipeline'):
        out_bgr = apply_pipeline(bgr, pipeline)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        show_before_after(img_rgb, out_rgb, 'الأصلية', 'النتيجة النهائية')

        # download
        buf = save_image_to_bytes(out_bgr, fmt='PNG')
        st.download_button("تحميل الصورة النهائية (PNG)", data=buf, file_name='result.png', mime='image/png')

# --------------------- Dispatcher ---------------------

lecture_funcs = {
    lectures[0]: lecture1,
    lectures[1]: lecture2,
    lectures[2]: lecture3,
    lectures[3]: lecture4,
    lectures[4]: lecture5,
    lectures[5]: lecture6,
    lectures[6]: lecture7,
    lectures[7]: lecture8,
    lectures[8]: lecture9,
}

# run selected lecture
lecture_funcs[choice](orig_rgb)

st.sidebar.markdown("---")
st.sidebar.write("بُني بواسطة: محمد — مشروع تعليمي لمادة معالجة الصور")

# Footer
st.write("\n---\n*ملاحظة:* هذه نسخة أولية. يمكن تحسين الواجهة وتصغير/تنظيم الأكواد إلى ملفات منفصلة `modules/` لو أردت تحويلها لمشروع إنتاجي.")
