"""Compat layer (legacy imports)."""

from app.vision.face_detection import FaceDetector


def detect_faces(image_uploaded: bytes, output_img_dir: str):
    detector = FaceDetector()
    img = detector.load_image_from_bytes(image_uploaded)
    faces = detector.detect(img)
    return {"results": [(f.image, f.bbox) for f in faces], "img": img}


def path_to_images(images_dir, images: list) -> list[str]:
    return [f"{images_dir}/{img}" for img in images]
