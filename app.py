import io
import json
import tempfile
from pathlib import Path

import numpy as np
import requests
import streamlit as st
import torch
import timm

from PIL import Image
from torchvision import transforms

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


# --------------------------------------------------
# إعداد التطبيق
# --------------------------------------------------

st.set_page_config(
    page_title="البحث البصري في العمارة الإيلخانية",
    layout="wide",
)

MODEL_NAME = "vit_base_patch16_224"

MODEL_URL = (
    "https://github.com/islamali664011/"
    "ilkhanid-image-features-Architecture-and-Arts-/"
    "releases/download/v1.0.0/vit_ilkhanid.pt"
)

DRIVE_FOLDER_ID = "10LBW8ZrZ_1xfkEn3F7iodpJ2AZ6YVUNt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

CACHE_FOLDER = Path(tempfile.gettempdir()) / "ilkhanid_cache"
CACHE_FOLDER.mkdir(parents=True, exist_ok=True)

MODEL_PATH = CACHE_FOLDER / "vit_ilkhanid.pt"


# --------------------------------------------------
# تحميل الموديل من GitHub Releases
# --------------------------------------------------

@st.cache_resource(show_spinner="جارٍ تحميل نموذج ViT...")
def load_model():
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 100_000_000:
        temporary_path = MODEL_PATH.with_suffix(".download")

        try:
            with requests.get(
                MODEL_URL,
                stream=True,
                timeout=(30, 300),
            ) as response:
                response.raise_for_status()

                with open(temporary_path, "wb") as output:
                    for chunk in response.iter_content(
                        chunk_size=1024 * 1024
                    ):
                        if chunk:
                            output.write(chunk)

            if temporary_path.stat().st_size < 100_000_000:
                raise RuntimeError(
                    "حجم ملف الموديل الذي تم تنزيله غير صحيح."
                )

            temporary_path.replace(MODEL_PATH)

        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    model = timm.create_model(
        MODEL_NAME,
        pretrained=False,
    )

    model.head = torch.nn.Identity()

    try:
        state = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        state = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()

    config = getattr(model, "pretrained_cfg", None)

    if not config:
        config = getattr(model, "default_cfg", {})

    mean = config.get("mean", (0.485, 0.456, 0.406))
    std = config.get("std", (0.229, 0.224, 0.225))

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return model, preprocess


# --------------------------------------------------
# الاتصال الآمن بـ Google Drive
# --------------------------------------------------

@st.cache_resource
def get_drive_service():
    if "service_account_json" not in st.secrets:
        raise RuntimeError(
            "بيانات حساب الخدمة غير موجودة في Streamlit Secrets."
        )

    service_account_info = json.loads(
        st.secrets["service_account_json"]
    )

    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=[
            "https://www.googleapis.com/auth/drive.readonly"
        ],
    )

    return build(
        "drive",
        "v3",
        credentials=credentials,
        cache_discovery=False,
    )


# --------------------------------------------------
# جلب الصور من مجلد Drive الخاص
# --------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def list_drive_images(folder_id, _service):
    files = []
    page_token = None

    while True:
        result = _service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields=(
                "nextPageToken,files("
                "id,name,mimeType,size,modifiedTime)"
            ),
            pageSize=1000,
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()

        for file_info in result.get("files", []):
            name = file_info.get("name", "")
            mime_type = file_info.get("mimeType", "")

            if (
                Path(name).suffix.lower() in IMAGE_EXTENSIONS
                and mime_type.startswith("image/")
            ):
                files.append(file_info)

        page_token = result.get("nextPageToken")

        if not page_token:
            break

    return sorted(
        files,
        key=lambda item: item.get("name", "").lower(),
    )


# --------------------------------------------------
# تنزيل صورة مرجعية إلى ذاكرة التخزين المؤقت
# --------------------------------------------------

def download_drive_image(file_info, service):
    file_id = file_info["id"]
    extension = Path(file_info["name"]).suffix.lower()

    local_path = CACHE_FOLDER / f"{file_id}{extension}"
    metadata_path = CACHE_FOLDER / f"{file_id}.meta"

    expected_metadata = (
        f"{file_info.get('size', '')}|"
        f"{file_info.get('modifiedTime', '')}"
    )

    if (
        local_path.exists()
        and metadata_path.exists()
        and metadata_path.read_text() == expected_metadata
    ):
        return local_path

    temporary_path = local_path.with_suffix(
        local_path.suffix + ".download"
    )

    request = service.files().get_media(fileId=file_id)

    try:
        with open(temporary_path, "wb") as output:
            downloader = MediaIoBaseDownload(
                output,
                request,
                chunksize=1024 * 1024,
            )

            done = False

            while not done:
                _, done = downloader.next_chunk()

        temporary_path.replace(local_path)
        metadata_path.write_text(expected_metadata)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return local_path


# --------------------------------------------------
# استخراج خصائص الصور وبناء فهرس البحث
# --------------------------------------------------

def extract_features(image, model, preprocess):
    image = image.convert("RGB")
    tensor = preprocess(image).unsqueeze(0)

    with torch.inference_mode():
        features = model(tensor)

    return features.squeeze(0).cpu().numpy().astype(np.float32)


@st.cache_data(show_spinner="جارٍ تجهيز الصور المرجعية...")
def build_image_index(
    signatures,
    _service,
    _model,
    _preprocess,
):
    image_paths = []
    feature_vectors = []

    for file_info in signatures:
        try:
            path = download_drive_image(file_info, _service)

            with Image.open(path) as image:
                vector = extract_features(
                    image,
                    _model,
                    _preprocess,
                )

            image_paths.append(str(path))
            feature_vectors.append(vector)

        except Exception:
            continue

    if not feature_vectors:
        return [], np.empty((0, 768), dtype=np.float32)

    return image_paths, np.vstack(feature_vectors)


# --------------------------------------------------
# واجهة المستخدم
# --------------------------------------------------

st.title("البحث البصري في العمارة والفنون الإيلخانية")

st.write(
    "ارفع صورة معمارية أو زخرفية، وسيبحث التطبيق عن "
    "أقرب الصور بصريًا داخل مجموعتك المرجعية."
)

st.caption(
    "النموذج: Vision Transformer (ViT) — "
    "المقارنة باستخدام التشابه الكوني."
)

try:
    service = get_drive_service()

    with st.spinner("جارٍ الاتصال بمجلد الصور الخاص..."):
        drive_images = list_drive_images(
            DRIVE_FOLDER_ID,
            service,
        )

    if not drive_images:
        st.error(
            "لم يتم العثور على صور. تأكد من مشاركة مجلد الصور "
            "مع حساب الخدمة بصلاحية Viewer."
        )
        st.stop()

    model, preprocess = load_model()

    signatures = tuple(
        (
            item["id"],
            item["name"],
            item.get("mimeType", ""),
            item.get("size", ""),
            item.get("modifiedTime", ""),
        )
        for item in drive_images
    )

    image_lookup = {
        item["id"]: item for item in drive_images
    }

    index_signatures = tuple(
        image_lookup[item[0]] for item in signatures
    )

    reference_paths, reference_features = build_image_index(
        index_signatures,
        service,
        model,
        preprocess,
    )

except Exception as error:
    st.error("تعذر تشغيل التطبيق أو الوصول إلى الملفات.")
    st.code(str(error))
    st.stop()


if len(reference_paths) == 0:
    st.error("لم نتمكن من تجهيز الصور المرجعية للبحث.")
    st.stop()


st.success(
    f"تم تجهيز {len(reference_paths)} صورة مرجعية للبحث."
)

uploaded_file = st.file_uploader(
    "اختر صورة للبحث عن صور مشابهة",
    type=["jpg", "jpeg", "png"],
)

top_k = st.slider(
    "عدد النتائج",
    min_value=1,
    max_value=min(10, len(reference_paths)),
    value=min(5, len(reference_paths)),
)

if uploaded_file is not None:
    try:
        query_image = Image.open(
            io.BytesIO(uploaded_file.getvalue())
        ).convert("RGB")

        st.image(
            query_image,
            caption="الصورة المدخلة",
            width=350,
        )

        if st.button("ابحث عن الصور المشابهة"):
            query_vector = extract_features(
                query_image,
                model,
                preprocess,
            )

            query_norm = np.linalg.norm(query_vector)
            reference_norms = np.linalg.norm(
                reference_features,
                axis=1,
            )

            denominator = np.maximum(
                reference_norms * query_norm,
                1e-12,
            )

            scores = (
                reference_features @ query_vector
            ) / denominator

            ranked_indices = np.argsort(scores)[::-1][:top_k]

            st.subheader("أقرب الصور إلى الصورة المدخلة")

            columns = st.columns(
                min(3, len(ranked_indices))
            )

            for rank, index in enumerate(ranked_indices):
                path = reference_paths[index]

                score = float(scores[index])

                with columns[rank % len(columns)]:
                    st.image(
                        path,
                        caption=(
                            f"النتيجة {rank + 1} — "
                            f"التشابه: {score:.4f}"
                        ),
                        width="stretch",
                    )

                    st.caption(Path(path).name)

    except Exception as error:
        st.error("حدث خطأ أثناء البحث عن الصور.")
        st.code(str(error))
