import os
import base64
import re  # <-- Added for text cleaning
from groq import Groq
from src.config import settings


class MultimodalVisionEngine:
    def __init__(self):
        """
        Initializes the cloud-based Groq client using your environment settings.
        """
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model_name = settings.GROQ_VISION_MODEL

    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Converts a local image file into a base64 string for API transmission.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _sanitize_response(self, text: str) -> str:
        """
        Removes internal reasoning tokens like <think>...</think> from the LLM output.
        """
        if not text:
            return ""
        # Regex explanation: re.DOTALL makes sure the dot matches newlines too.
        # This strips out <think> and everything inside it up to </think>.
        cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return cleaned_text.strip()

    def describe_image(self, image_path: str) -> str:
        """
        Sends a cached image to Groq's LPU cloud infrastructure for rapid technical analysis.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at cache target: {image_path}")

        print(f"[Vision] Streaming image to Groq LPU ({self.model_name}): {os.path.basename(image_path)}...")

        base64_image = self._encode_image_to_base64(image_path)

        prompt = (
            "You are an expert technical data analyst. Analyze this diagram, chart, or image "
            "extracted from a document. Provide a comprehensive, highly detailed description of "
            "everything it contains. If it is a flowchart, graph, or system architecture, explain every step, "
            "connection, and label precisely. Do not speak conversationally; respond only with "
            "the clean technical description of the contents."
        )

        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0.2
            )

            raw_description = chat_completion.choices[0].message.content

            clean_description = self._sanitize_response(raw_description)
            return clean_description if clean_description else "[Empty response after sanitization]"

        except Exception as e:
            print(f"[Vision Error] Groq API call failed: {e}")
            return f"[Failure during cloud visual analysis phase of: {os.path.basename(image_path)}]"