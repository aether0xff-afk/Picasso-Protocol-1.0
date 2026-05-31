from pathlib import Path
import sys

import torch
import tkinter as tk
from tkinter import filedialog, messagebox, font
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from picasso_protocol.image_codec import (  # noqa: E402
    combine_jamo,
    decode_data_from_image,
    encode_data_to_image,
)
from picasso_protocol.models import ArtistY  # noqa: E402

class AppY_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Artist Y - The Re-interpreter")
        self.root.geometry("600x550")
        self.root.configure(bg="#2E2E2E")
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.font_title = font.Font(family="Helvetica", size=18, weight="bold")
        self.font_main = font.Font(family="Helvetica", size=10)
        self.font_status = font.Font(family="Helvetica", size=9)
        self.bg_color = "#2E2E2E"
        self.fg_color = "#FFFFFF"
        self.entry_bg = "#3C3C3C"
        self.button_bg = "#4A4A4A"
        self.button_active_bg = "#5A5A5A"
        
        self.create_widgets()

    def create_widgets(self):
        main_frame = tk.Frame(self.root, bg=self.bg_color, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(main_frame, text="Artist Y", font=self.font_title, bg=self.bg_color, fg=self.fg_color).pack(pady=(0, 20))
        
        self.load_button = tk.Button(main_frame, text="Artist Y 모델 불러오기 (.pth)", command=self.load_model, 
                                     bg=self.button_bg, fg=self.fg_color, activebackground=self.button_active_bg, 
                                     activeforeground=self.fg_color, borderwidth=0, font=self.font_main, relief=tk.FLAT, padx=10, pady=5)
        self.load_button.pack(pady=10, fill=tk.X)

        tk.Label(main_frame, text="텍스트 입력:", font=self.font_main, bg=self.bg_color, fg=self.fg_color, anchor='w').pack(fill=tk.X, pady=(10, 5))
        self.text_input = tk.Text(main_frame, height=5, width=60, bg=self.entry_bg, fg=self.fg_color, 
                                  insertbackground=self.fg_color, selectbackground="#5A5A5A", borderwidth=0, font=self.font_main)
        self.text_input.pack(fill=tk.X)

        button_frame = tk.Frame(main_frame, bg=self.bg_color)
        button_frame.pack(pady=20, fill=tk.X)
        
        self.encode_button = tk.Button(button_frame, text="텍스트 → PNG 이미지", command=self.encode_text, state=tk.DISABLED,
                                     bg=self.button_bg, fg=self.fg_color, activebackground=self.button_active_bg, 
                                     activeforeground=self.fg_color, borderwidth=0, font=self.font_main, relief=tk.FLAT, padx=10, pady=5)
        self.encode_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        
        self.decode_button = tk.Button(button_frame, text="PNG 이미지 → 텍스트 재해석", command=self.decode_image, state=tk.DISABLED,
                                     bg=self.button_bg, fg=self.fg_color, activebackground=self.button_active_bg, 
                                     activeforeground=self.fg_color, borderwidth=0, font=self.font_main, relief=tk.FLAT, padx=10, pady=5)
        self.decode_button.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(5, 0))

        tk.Label(main_frame, text="재해석 결과:", font=self.font_main, bg=self.bg_color, fg=self.fg_color, anchor='w').pack(fill=tk.X, pady=(10, 5))
        self.text_output = tk.Text(main_frame, height=5, width=60, state=tk.DISABLED, bg=self.entry_bg, fg=self.fg_color, 
                                   borderwidth=0, font=self.font_main)
        self.text_output.pack(fill=tk.X)

        self.status_label = tk.Label(self.root, text="모델을 불러와주세요.", bd=1, relief=tk.SUNKEN, anchor=tk.W, 
                                     font=self.font_status, bg="#3C3C3C", fg="#A0A0A0")
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def load_model(self):
        filepath = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")], title="artist_y_standalone.pth 파일을 선택하세요")
        if not filepath: return
        
        try:
            self.model = ArtistY()
            self.model.load_state_dict(torch.load(filepath, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            self.status_label.config(text=f"✅ 모델 로딩 완료. (Device: {str(self.device).upper()})")
            self.encode_button.config(state=tk.NORMAL)
            self.decode_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("모델 로딩 오류", f"모델을 불러오는 중 오류가 발생했습니다: {e}")
            self.model = None

    def encode_text(self):
        if not self.model: return
        input_text = self.text_input.get("1.0", "end-1c").strip()
        if not input_text: messagebox.showwarning("입력 오류", "텍스트를 입력해주세요."); return
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")], title="추상화 PNG 이미지로 저장")
        if not filepath: return
        with torch.no_grad():
            inputs = self.model.tokenizer(input_text, return_tensors='pt', max_length=128, padding='max_length', truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            latent_vector = self.model.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask']).last_hidden_state
            original_length = int(torch.sum(inputs['attention_mask']).item())
        image = encode_data_to_image(latent_vector, original_length)
        image.save(filepath, "PNG")
        self.status_label.config(text=f"✅ '{filepath}'에 PNG 이미지 저장 완료.")
        messagebox.showinfo("성공", f"'{filepath}'에 PNG 이미지를 저장했습니다.")

    def decode_image(self):
        if not self.model: return
        filepath = filedialog.askopenfilename(filetypes=[("PNG Image", "*.png")], title="재해석할 PNG 이미지 열기")
        if not filepath: return
        try:
            image = Image.open(filepath)
            latent_vector, _ = decode_data_from_image(image)
        except Exception as e:
            messagebox.showerror("파일 오류", f"이미지 파일을 처리하는 중 오류가 발생했습니다: {e}"); return
        with torch.no_grad():
            latent_vector = latent_vector.to(self.device)
            attention_mask = torch.ones(latent_vector.shape[:-1], dtype=torch.long).to(self.device)
            
            outputs = self.model.decoder.generate(
                inputs_embeds=latent_vector, attention_mask=attention_mask,
                max_new_tokens=50, do_sample=True, top_k=50,
                pad_token_id=self.model.tokenizer.pad_token_id
            )
            raw_text = self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            reinterpreted_text = combine_jamo(raw_text)
            
        self.text_output.config(state=tk.NORMAL)
        self.text_output.delete("1.0", tk.END)
        self.text_output.insert("1.0", reinterpreted_text)
        self.text_output.config(state=tk.DISABLED)
        self.status_label.config(text=f"✅ '{filepath}' 파일 재해석 완료.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AppY_GUI(root)
    root.mainloop()
