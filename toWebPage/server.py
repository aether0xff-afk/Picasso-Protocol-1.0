from pathlib import Path
import sys

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import base64
import io
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from picasso_protocol.image_codec import (  # noqa: E402
    combine_jamo,
    decode_data_from_image,
    encode_data_to_image,
)
from picasso_protocol.model_paths import resolve_model_path  # noqa: E402
from picasso_protocol.models import ArtistX, ArtistY  # noqa: E402


# --- Flask 앱 설정 ---
app = Flask(__name__)
CORS(app)

# --- 모델 로딩 ---
print("서버 시작 중... 모델을 불러옵니다.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_x = ArtistX()
model_x.load_state_dict(
    torch.load(
        resolve_model_path("artist_x_best_model.pth", project_root=ROOT_DIR),
        map_location=device,
    )
)
model_x.to(device).eval()

model_y = ArtistY()
model_y.load_state_dict(
    torch.load(
        resolve_model_path("artist_y_standalone.pth", project_root=ROOT_DIR),
        map_location=device,
    )
)
model_y.to(device).eval()

print(f"✅ 모델 로딩 완료. (Device: {device})")


# --- API 엔드포인트 ---
@app.route('/encode', methods=['POST'])
def encode():
    payload = request.get_json(silent=True) or {}
    text: Optional[str] = payload.get('text')
    mode: str = payload.get('mode', 'y')

    if not text:
        return jsonify({"error": "텍스트가 없습니다."}), 400

    model = model_y if mode == 'y' else model_x

    with torch.no_grad():
        inputs = model.tokenizer(
            text,
            return_tensors='pt',
            max_length=128,
            padding='max_length',
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        enc_output = model.encoder(
            inputs['input_ids'], attention_mask=inputs['attention_mask']
        )
        latent_vector = enc_output.last_hidden_state
        # .item() 은 Python number 반환, int() 로 명시적 보장
        original_length = int(torch.sum(inputs['attention_mask']).item())

    image = encode_data_to_image(latent_vector, original_length)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_str = base64.b64encode(buf.getvalue()).decode()

    return jsonify({"image": img_str})


@app.route('/decode', methods=['POST'])
def decode():
    payload = request.get_json(silent=True) or {}
    base64_image: Optional[str] = payload.get('image')
    mode: str = payload.get('mode', 'y')

    if not base64_image:
        return jsonify({"error": "이미지 데이터가 없습니다."}), 400

    try:
        raw = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(raw))
        latent_vector, original_length = decode_data_from_image(image)
    except Exception as e:
        return jsonify({"error": f"이미지 처리 오류: {e}"}), 400

    model = model_y if mode == 'y' else model_x
    latent_vector = latent_vector.to(device)

    with torch.no_grad():
        attention_mask = torch.ones(latent_vector.shape[:-1], dtype=torch.long).to(device)
        if mode == 'x':
            logits = model.decoder(
                inputs_embeds=latent_vector, attention_mask=attention_mask
            ).logits
            pred_ids = torch.argmax(logits, dim=-1)
            clean_ids = pred_ids[0][:original_length]
            raw_text = model.tokenizer.decode(clean_ids, skip_special_tokens=True)
            result_text = combine_jamo(raw_text)
        else:  # mode 'y'
            outputs = model.decoder.generate(
                inputs_embeds=latent_vector,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=True,
                top_k=50,
                pad_token_id=model.tokenizer.pad_token_id,
            )
            raw_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            result_text = combine_jamo(raw_text)

    return jsonify({"text": result_text})


if __name__ == '__main__':
    app.run(debug=True)
