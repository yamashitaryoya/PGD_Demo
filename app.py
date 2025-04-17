# ===============================
# 1. 必要ライブラリのインポート
# ===============================
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import foolbox as fb
import urllib.request
import io

# ===============================
# 2. タイトル・説明
# ===============================
st.title("🔒 PGD攻撃デモ")
#st.write("画像をアップロードし、PGD攻撃により分類がどのように変化するかを確認できます。")

# ===============================
# 3. 画像アップロード
# ===============================
uploaded_file = st.file_uploader("画像をアップロードしてください（JPEG/PNG）", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # ===============================
    # 4. 前処理
    # ===============================
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0)

    # ===============================
    # 5. モデルとデバイス
    # ===============================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=True).to(device).eval()
    input_tensor = input_tensor.to(device)
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device)

    # ===============================
    # 6. クラスラベルの読み込み
    # ===============================
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    categories = urllib.request.urlopen(url).read().decode("utf-8").splitlines()

    # ===============================
    # 7. 元ラベルの予測
    # ===============================
    logits = model(input_tensor)
    label = logits.argmax(dim=1).item()
    label_name = categories[label]

    # ===============================
    # 8. PGD攻撃
    # ===============================
    attack = fb.attacks.LinfPGD(steps=40, rel_stepsize=0.05, random_start=True)
    epsilons = 0.03
    raw_adv, clipped_adv, is_adv = attack(fmodel, input_tensor, torch.tensor([label]).to(device), epsilons=epsilons)

    # 敵対ラベルの取得
    adv_output = model(clipped_adv)
    adv_label = adv_output.argmax(dim=1).item()
    adv_label_name = categories[adv_label]

    # ===============================
    # 9. ノイズ画像の可視化
    # ===============================
    noise = clipped_adv - input_tensor
    noise_abs = (noise - noise.min()) / (noise.max() - noise.min())

    # ===============================
    # 10. 表示関数
    # ===============================
    def tensor_to_img(tensor):
        img = tensor.squeeze().detach().cpu().permute(1, 2, 0).numpy()
        return np.clip(img, 0, 1)

    # ===============================
    # 🔍 11. 結果表示
    # ===============================
    st.subheader("🎯 分類結果")
    st.write(f"**元のラベル:** {label_name}")
    st.write(f"**攻撃後ラベル:** {adv_label_name} （成功: {bool(is_adv.item())}）")

    st.subheader("🖼️ 画像表示")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(tensor_to_img(input_tensor), caption=f"Original: {label_name}", use_container_width=True)

    with col2:
        st.image(tensor_to_img(clipped_adv), caption=f"PGD Adversarial: {adv_label_name}", use_container_width=True)

    with col3:
        st.image(tensor_to_img(noise_abs), caption="Noise", use_container_width=True)
