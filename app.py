from pathlib import Path

import streamlit as st
import torch
import timm
import numpy as np

from PIL import Image
from torchvision import transforms


# إعداد التطبيق
st.set_page_config(
    page_title="البحث البصري في العمارة الإيلخانية",
    page_icon=None,
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
IMAGE_FOLDER = ROOT / "reference_images"
MODEL_PATH = ROOT / "vit_ilkhanid.pt"

MODEL_NAME = "vit_base_patch16_224"
EXTENSIONS = {".jpg", ".jpeg", ".png"}


# تحميل نموذج ViT
@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        model = timm.create_model(
            MODEL_NAME,
            pretrained=False
        )
    else:
        model = timm.create_model(
            MODEL_NAME,
            pretrained=True
        )

    model.head = torch.nn.Identity()

    if MODEL_PATH.exists():
        try:
            state = torch.load(
                MODEL_PATH,
                map_location="cpu",
                weights_only=True
            )
        except TypeError:
            state = torch.load(
                MODEL_PATH,
                map_location="cpu"
            )

        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        model.load_state_dict(state)

    model.eval()

    cfg = getattr(model, "default_cfg", {})
    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return model, preprocess


# استخراج خصائص الصورة
def extract_features(image, model, preprocess):
    image = image.convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    with torch.inference_mode():
        features = model(tensor)

    return features.squeeze(0).cpu().numpy().astype(np.float32)


# إعداد قاعدة الصور المرجعية
@st.cache_data(show_spinner="جارٍ تحليل الصور المرجعية...")
def build_image_index(file_signatures, _model, _preprocess):
    valid_paths = []
    feature_vectors = []

    for path, _, _ in file_signatures:
        try:
            with Image.open(path) as image:
                vector = extract_features(
                    image, _model, _preprocess
                )

            valid_paths.append(path)
            feature_vectors.append(vector)

        except Exception:
            continue

    if not feature_vectors:
        return [], np.empty((0, 768), dtype=np.float32)

    return valid_paths, np.vstack(feature_vectors)


# واجهة التطبيق
st.title("البحث البصري في العمارة والفنون الإيلخانية")

st.write(
    "ارفع صورة معمارية أو زخرفية للعثور على أكثر الصور "
    "تشابهًا معها داخل قاعدة الصور المرجعية."
)

st.caption(
    "يعتمد البحث على خصائص الصور المستخرجة بواسطة "
    "Vision Transformer (ViT) ومقياس التشابه الكوني."
)

# التأكد من وجود قاعدة الصور
if not IMAGE_FOLDER.exists():
    st.warning(
        "قاعدة الصور المرجعية غير موجودة بعد. "
        "يجب إضافة مجلد reference_images إلى مستودع GitHub "
        "ووضع صور العمارة والفنون الإيلخانية بداخله."
    )
    st.stop()

image_files = sorted(
    path for path in IMAGE_FOLDER.iterdir()
    if path.is_file() and path.suffix.lower() in EXTENSIONS
)

if not image_files:
    st.warning(
        "لم يتم العثور على صور مرجعية. "
        "أضف صور JPG أو JPEG أو PNG إلى مجلد reference_images."
    )
    st.stop()

try:
    model, preprocess = load_model()

    signatures = tuple(
        (
            str(path),
            path.stat().st_size,
            path.stat().st_mtime_ns
        )
        for path in image_files
    )

    reference_paths, reference_features = build_image_index(
        signatures, model, preprocess
    )

except Exception as error:
    st.error(f"تعذر تجهيز النموذج أو قاعدة الصور: {error}")
    st.stop()

if not reference_paths:
    st.error("تعذر استخراج خصائص الصور المرجعية.")
    st.stop()

st.success(
    f"قاعدة الصور جاهزة: {len(reference_paths)} صورة."
)

uploaded_file = st.file_uploader(
    "ارفع الصورة المطلوب البحث عن صور مشابهة لها",
    type=["jpg", "jpeg", "png"]
)

max_results = min(10, len(reference_paths))

top_k = st.slider(
    "عدد الصور المشابهة المطلوبة",
    min_value=1,
    max_value=max_results,
    value=min(5, max_results)
)

if uploaded_file is not None:
    try:
        with Image.open(uploaded_file) as image:
            query_image = image.convert("RGB")

        st.image(
            query_image,
            caption="الصورة المدخلة",
            width=350
        )

        if st.button("ابحث عن الصور المشابهة"):
            query_vector = extract_features(
                query_image, model, preprocess
            )

            query_norm = np.linalg.norm(query_vector)
            reference_norms = np.linalg.norm(
                reference_features, axis=1
            )

            denominator = np.maximum(
                reference_norms * query_norm, 1e-12
            )

            scores = (
                reference_features @ query_vector
            ) / denominator

            ranked_indices = np.argsort(scores)[::-1][:top_k]

            st.subheader("الصور الأكثر تشابهًا")

            columns = st.columns(min(3, len(ranked_indices)))

            for rank, index in enumerate(ranked_indices):
                path = reference_paths[index]
                score = float(scores[index])

                with columns[rank % len(columns)]:
                    st.image(
                        str(path),
                        caption=(
                            f"النتيجة {rank + 1} | "
                            f"درجة التشابه: {score:.4f}"
                        ),
                        width="stretch"
                    )

                    st.caption(path.name)

    except Exception as error:
        st.error(f"حدث خطأ أثناء معالجة الصورة: {error}")
