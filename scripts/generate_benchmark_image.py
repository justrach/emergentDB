#!/usr/bin/env python3
"""
Generate a professional benchmark visualization image using Gemini.
"""

import base64
import mimetypes
import os
from google import genai
from google.genai import types


def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    print(f"File saved to: {file_name}")


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    prompt = """Create a horizontal bar chart comparing vector database search latency.

Title: "EmergentDB vs Competitors - 10K Vectors"
Subtitle: "Search Latency (microseconds) | Lower = Faster"

EXACT DATA - 4 horizontal bars from top to bottom:

1. "EmergentDB (m=8)" - 44μs - Use bright GREEN (#22c55e)
2. "EmergentDB (m=16)" - 102μs - Use GREEN (#16a34a)
3. "ChromaDB" - 2,259μs - Use ORANGE (#f59e0b)
4. "LanceDB" - 3,590μs - Use PURPLE (#8b5cf6)

IMPORTANT: The EmergentDB bars should be VERY SHORT (44 and 102) compared to ChromaDB (2,259) and LanceDB (3,590). This is the key visual - EmergentDB is 51x faster!

Add these labels:
- Show "51x faster" with an arrow pointing from ChromaDB to EmergentDB
- Show recall percentages: EmergentDB 100%, ChromaDB 99.8%, LanceDB 84.3%

Style:
- Dark background (#111827)
- White/light text
- Clean, modern, minimal design
- Wide format (like 1200x500)
- Database names on the left of each bar
- Latency values at the end of each bar"""

    model = "gemini-2.0-flash-exp-image-generation"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        response_modalities=[
            "IMAGE",
            "TEXT",
        ],
    )

    output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    public_dir = os.path.join(output_dir, "public")
    os.makedirs(public_dir, exist_ok=True)

    file_index = 0
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        if chunk.candidates[0].content.parts[0].inline_data and chunk.candidates[0].content.parts[0].inline_data.data:
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)

            if file_index == 0:
                file_name = os.path.join(public_dir, f"benchmark{file_extension}")
            else:
                file_name = os.path.join(public_dir, f"benchmark_{file_index}{file_extension}")

            save_binary_file(file_name, data_buffer)
            file_index += 1
        else:
            print(chunk.text)

    if file_index == 0:
        print("No image was generated. Try running again.")
    else:
        print(f"\nGenerated {file_index} image(s) in {public_dir}/")


if __name__ == "__main__":
    generate()
