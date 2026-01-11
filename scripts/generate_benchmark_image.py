#!/usr/bin/env python3
"""
Generate a professional benchmark visualization image using Gemini 3 Pro.
"""

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

    prompt = """Create a stunning, professional benchmark comparison infographic for EmergentDB vector database.

TITLE: "EmergentDB Performance Benchmark"
SUBTITLE: "10,000 Vectors | 768 Dimensions | Real Gemini Embeddings | Proof of Concept"

THE DATA (show as horizontal bar chart):
- EmergentDB (HNSW m=8): 44 microseconds latency, 100% recall - BRIGHT GREEN
- EmergentDB (HNSW m=16): 102 microseconds latency, 100% recall - GREEN
- ChromaDB (HNSW): 2,259 microseconds (2.26ms) latency, 99.8% recall - ORANGE
- LanceDB (IVF-PQ): 3,590 microseconds (3.59ms) latency, 84.3% recall - PURPLE

KEY VISUAL: The EmergentDB bars should be DRAMATICALLY shorter than ChromaDB and LanceDB to show the 51x and 82x speed advantage. This contrast is the main point!

CALLOUTS TO INCLUDE:
- "51x FASTER" comparing EmergentDB to ChromaDB
- "82x FASTER" comparing EmergentDB to LanceDB
- Show recall percentages as badges or labels

DESIGN STYLE:
- Dark gradient background (dark blue/slate like #0f172a to #1e293b)
- Modern, clean, minimalist tech aesthetic
- Similar to Vercel, Linear, or Stripe marketing graphics
- High contrast, easy to read
- Professional typography
- Subtle grid lines or visual guides

IMPORTANT DETAILS:
- Label each database clearly with its name AND index type
- Show exact latency values on or near each bar
- Include "Recall@10" percentages
- Add note: "Lower latency = Better | Tested with real Gemini embeddings"
- Make it look like a premium tech company's marketing material
- Include "PROOF OF CONCEPT" badge or label somewhere

Make this image absolutely stunning and clear - it will be the hero image on a GitHub README. Make it wide format (landscape) suitable for a README header."""

    model = "models/gemini-3-pro-image-preview"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        response_modalities=["IMAGE", "TEXT"],
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
