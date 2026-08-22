"""Plant Leaf Disease Detection and Bangla Agricultural Advisory Service."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..core.logger import get_logger

logger = get_logger("vision.plant")


@dataclass
class PlantAnalysisResult:
    plant_name: str
    disease_name: str
    disease_name_bn: str
    confidence: float
    symptoms_bn: str
    cause_bn: str
    remedy_bn: str
    is_healthy: bool = False
    confidence_is_low: bool = False


# Known plant disease database with natural Bangla agricultural explanations
PLANT_DISEASE_DB: Dict[str, Dict[str, Any]] = {
    "tomato_early_blight": {
        "plant": "Tomato (টমেটো)",
        "disease_en": "Early Blight",
        "disease_bn": "আগাম ধ্বসা রোগ (Early Blight)",
        "symptoms_bn": "পাতার নিচের অংশে বাদামী বা কালচে দাগ এবং পাতার কিনারা হলুদ হয়ে শুকিয়ে যাওয়া।",
        "cause_bn": "অল্টারনারিয়া সোলানি (Alternaria solani) নামক ছত্রাকের আক্রমণ এবং অতিরিক্ত আর্দ্রতা।",
        "remedy_bn": "আক্রান্ত পাতাগুলো সাবধানে কেটে পুড়িয়ে ফেলুন। গাছে ট্রাইকোডার্মা বা অনুমোদিত কপার ছত্রাকনাশক স্প্রে করুন এবং গাছের গোড়ায় পানি জমতে দেবেন না।",
    },
    "potato_late_blight": {
        "plant": "Potato (আলু)",
        "disease_en": "Late Blight",
        "disease_bn": "নাবি ধ্বসা রোগ (Late Blight)",
        "symptoms_bn": "পাতায় ভেজা ভেজা কালচে দাগ এবং পাতার নিচের দিকে সাদাটে ছত্রাকের আবরণ তৈরি হওয়া।",
        "cause_bn": "ফাইটোফথোরা ইনফেস্টানস (Phytophthora infestans) ছত্রাক।",
        "remedy_bn": "দ্রুত মেনকোজেব বা ম্যানকোজেব জাতীয় ছত্রাকনাশক কুয়াশাচ্ছন্ন আবহাওয়ায় স্প্রে করুন এবং আক্রান্ত গাছ আলাদা করুন।",
    },
    "healthy_leaf": {
        "plant": "General Plant (গাছ)",
        "disease_en": "Healthy",
        "disease_bn": "সুস্থ ও সতেজ পাতা",
        "symptoms_bn": "পাতায় কোনো রোগ বা ছত্রাকের লক্ষণ দেখা যাচ্ছে না।",
        "cause_bn": "গাছের পুষ্টি ও পরিবেশ স্বাভাবিক আছে।",
        "remedy_bn": "নিয়মিত পর্যাপ্ত সূর্যালোক ও পরিমিত পানি দিন।",
    },
}


class PlantDiseaseDetector:
    """Classifies plant leaf diseases and generates clear, cautious Bangla advice."""

    def __init__(self) -> None:
        self.model_path = os.path.join("data", "models", "plant_disease.tflite")

    def analyze_leaf(self, frame: Any) -> PlantAnalysisResult:
        """Process leaf image, perform classification, and return diagnosis."""
        if frame is None:
            return PlantAnalysisResult(
                plant_name="Unknown",
                disease_name="None",
                disease_name_bn="শনাক্ত করা যায়নি",
                confidence=0.0,
                symptoms_bn="ছবি পাওয়া যায়নি।",
                cause_bn="ক্যামেরা ফ্রেম অনুপস্থিত।",
                remedy_bn="ক্যামেরা দিয়ে পাতার পরিষ্কার ছবি দেখান।",
                confidence_is_low=True,
            )

        # 1. Try TFLite model inference
        try:
            if os.path.exists(self.model_path):
                import numpy as np
                import cv2
                
                try:
                    import tflite_runtime.interpreter as tflite
                except ImportError:
                    import tensorflow.lite as tflite

                interpreter = tflite.Interpreter(model_path=self.model_path)
                interpreter.allocate_tensors()

                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

                # Preprocess image
                input_shape = input_details[0]['shape']
                resized_img = cv2.resize(frame, (input_shape[1], input_shape[2]))
                input_data = np.expand_dims(resized_img, axis=0)
                input_data = (np.float32(input_data) - 127.5) / 127.5

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                output_data = interpreter.get_tensor(output_details[0]['index'])
                top_prediction = np.argmax(output_data)
                confidence = float(np.max(output_data))

                # Map to classes (assuming a simple map for demo)
                classes = list(PLANT_DISEASE_DB.keys())
                pred_class = classes[min(top_prediction, len(classes)-1)]
                
                info = PLANT_DISEASE_DB.get(pred_class, PLANT_DISEASE_DB["healthy_leaf"])
                return PlantAnalysisResult(
                    plant_name=info["plant"],
                    disease_name=info["disease_en"],
                    disease_name_bn=info["disease_bn"],
                    confidence=confidence,
                    symptoms_bn=info["symptoms_bn"],
                    cause_bn=info["cause_bn"],
                    remedy_bn=info["remedy_bn"],
                    is_healthy=(pred_class == "healthy_leaf"),
                    confidence_is_low=confidence < 0.65,
                )
        except Exception as e:
            logger.error(f"Error in TFLite inference: {e}")

        # 2. Cloud fallback: Google Gemini Vision API
        try:
            import cv2
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            
            if api_key:
                client = OpenAI(
                    api_key=api_key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
                
                _, buffer = cv2.imencode('.jpg', frame)
                base64_image = base64.b64encode(buffer).decode('utf-8')
                
                response = client.chat.completions.create(
                    model="gemini-1.5-flash",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Identify the plant disease in this image. Respond with a JSON object containing keys: 'disease_id' (one of: 'tomato_early_blight', 'potato_late_blight', 'healthy_leaf'), 'confidence' (float 0.0 to 1.0)."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                )
                
                response_text = response.choices[0].message.content
                import re
                json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(1))
                else:
                    data = json.loads(response_text)
                    
                disease_id = data.get("disease_id", "healthy_leaf")
                confidence = float(data.get("confidence", 0.9))
                
                info = PLANT_DISEASE_DB.get(disease_id, PLANT_DISEASE_DB["healthy_leaf"])
                return PlantAnalysisResult(
                    plant_name=info["plant"],
                    disease_name=info["disease_en"],
                    disease_name_bn=info["disease_bn"],
                    confidence=confidence,
                    symptoms_bn=info["symptoms_bn"],
                    cause_bn=info["cause_bn"],
                    remedy_bn=info["remedy_bn"],
                    is_healthy=(disease_id == "healthy_leaf"),
                    confidence_is_low=confidence < 0.65,
                )
        except Exception as e:
            logger.error(f"Error in Cloud Fallback: {e}")

        # 3. Complete failure fallback
        return PlantAnalysisResult(
            plant_name="Unknown",
            disease_name="None",
            disease_name_bn="বিশ্লেষণ সম্ভব নয়",
            confidence=0.0,
            symptoms_bn="বিশ্লেষণ করার জন্য প্রয়োজনীয় মডেল বা ইন্টারনেট সংযোগ নেই।",
            cause_bn="সিস্টেম ত্রুটি।",
            remedy_bn="ইন্টারনেট বা মডেল ফাইল পরীক্ষা করুন।",
            confidence_is_low=True,
        )

    def generate_bangla_speech_summary(self, result: PlantAnalysisResult) -> str:
        """Create conversational natural Bangla text for TTS explanation."""
        if result.confidence_is_low:
            return (
                f"আমি নিশ্চিত নই, তবে মনে হচ্ছে {result.plant_name} গাছে {result.disease_name_bn} হতে পারে। "
                "পাতার আরেকটি পরিষ্কার ছবি দেখালে ভালো হতো।"
            )

        if result.is_healthy:
            return f"আপনার {result.plant_name} পাতাটি সম্পূর্ণ সুস্থ ও সতেজ রয়েছে!"

        return (
            f"আমি পাতাটি পরীক্ষা করে দেখেছি। এটি সম্ভবত {result.plant_name} এর {result.disease_name_bn}। "
            f"লক্ষণ: {result.symptoms_bn} "
            f"পরামর্শ: {result.remedy_bn}"
        )
