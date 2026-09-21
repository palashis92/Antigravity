"""Enroll Face Utility Script for LUMI.

Allows capturing and saving face embeddings for Owner (Mizan) or any other person
directly into data/lumi.db so LUMI visually recognizes them immediately.

Usage on Raspberry Pi or PC:
    python3 scripts/enroll_face.py --name Mizan
    python3 scripts/enroll_face.py --name Palash --role creator
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lumi.memory.database import Database
from lumi.memory.manager import MemoryManager
from lumi.memory.models import ConsentStatus


def capture_face_embeddings(samples_needed: int = 3) -> list:
    """Capture camera frames and extract 128-dimensional face encodings."""
    try:
        import cv2
    except ImportError:
        print("[ERROR] OpenCV ('cv2') is required to capture face from camera. Please install opencv-python.")
        return []

    try:
        import face_recognition
        has_face_recognition = True
    except ImportError:
        has_face_recognition = False
        print("[WARNING] 'face_recognition' library is not installed. Encodings cannot be computed without it.")

    # Try camera opening (0, 1, or Picamera)
    cap = None
    for cam_idx in [0, 1, -1]:
        cap = cv2.VideoCapture(cam_idx)
        if cap.isOpened():
            print(f"[OK] Camera opened successfully on index {cam_idx}.")
            break

    if not cap or not cap.isOpened():
        print("[ERROR] Could not open camera. Please make sure the camera is connected and not locked by another process.")
        return []

    collected_embeddings = []
    print(f"\n[INFO] Look directly at the camera. Capturing {samples_needed} face samples...")
    start_t = time.time()

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    sample_count = 0
    while sample_count < samples_needed and (time.time() - start_t) < 30.0:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(faces) > 0:
            if has_face_recognition:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb)
                if encs:
                    collected_embeddings.append(encs[0].tolist())
                    sample_count += 1
                    print(f"  -> Captured sample {sample_count}/{samples_needed}")
                    time.sleep(0.6)  # Pause between samples for minor angle variation
            else:
                sample_count += 1
                print(f"  -> Face detected in frame {sample_count}/{samples_needed}")
                time.sleep(0.5)

    cap.release()
    return collected_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll face embeddings into LUMI database.")
    parser.add_argument("--name", type=str, default="Mizan", help="Name of the person (default: Mizan)")
    parser.add_argument("--role", type=str, default="owner", help="Relationship/role (owner, creator, friend, guest)")
    parser.add_argument("--samples", type=int, default=3, help="Number of face samples to capture (default: 3)")
    args = parser.parse_args()

    db_path = Path("data/lumi.db")
    db = Database(db_path=str(db_path), enable_wal=False)
    mem = MemoryManager(db)

    print(f"=== LUMI Face Enrollment: {args.name} ({args.role}) ===")
    embeddings = capture_face_embeddings(samples_needed=args.samples)

    person = mem.find_person_by_name(args.name)
    if not person:
        person = mem.remember_person(
            name=args.name,
            relationship=args.role,
            consent_status=ConsentStatus.GRANTED,
            notes=f"Registered via enroll_face.py ({args.role}).",
        )
        print(f"[OK] Created new profile for {args.name} (ID: {person.id}).")
    else:
        print(f"[OK] Found existing profile for {args.name} (ID: {person.id}).")

    if embeddings:
        for emb in embeddings:
            person.add_face_embedding(emb)
        mem.update_person(person)
        print(f"[SUCCESS] Saved {len(embeddings)} face embeddings for {person.name} in data/lumi.db!")
        print(f"[বাংলা] {person.name}-এর চেহারা সফলভাবে ডাটাবেজে সেভ হয়েছে। লুমি এখন তাকে সাথে সাথে চিনতে পারবে।")
    else:
        mem.update_person(person)
        print(f"[NOTE] Profile confirmed for {person.name} (without new face embeddings).")


if __name__ == "__main__":
    main()
