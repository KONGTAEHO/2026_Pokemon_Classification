import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torchvision.models as models
import torch.nn as nn

# 포켓몬 클래스 이름 불러오기
try:
    data_dir = "archive/PokemonData"
    class_names = sorted(os.listdir(data_dir))
except Exception:
    class_names = []

num_classes = max(len(class_names), 150)

@st.cache_resource
def load_model(model_path):
    # ResNet18 기반으로 데모 모델 가정
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        except Exception as e:
            st.warning("⚠️ 유효한 `best_model.pth` 파일을 불러올 수 없어 초기화된 모델을 사용합니다. 먼저 `train.py`를 실행하여 모델을 훈련시켜주세요.")
    model.eval()
    return model

# 최고 성능 모델 로드 (가정)
model_path = "best_model.pth"
model = load_model(model_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("Pokemon Classifier 🔴⚪")
st.write("포켓몬 이미지를 업로드하면 어떤 포켓몬인지 맞혀줍니다!")

uploaded_file = st.file_uploader("이미지 선택...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='업로드된 이미지', use_column_width=True)
    st.write("")
    
    if len(class_names) == 0:
        st.error("데이터셋(archive/PokemonData)을 찾을 수 없습니다. 클래스 정보를 불러오지 못했습니다.")
    else:
        st.write("분류 중...")
        
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_idx = predicted.item()
            confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted_idx].item()
            predicted_class_name = class_names[predicted_idx]
            
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"예측 결과: **{predicted_class_name}**")
            st.info(f"확률(신뢰도): {confidence:.2%}")
            
        with col2:
            try:
                class_dir = os.path.join(data_dir, predicted_class_name)
                if os.path.isdir(class_dir):
                    sample_images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                    if sample_images:
                        sample_img_path = os.path.join(class_dir, sample_images[0])
                        sample_image = Image.open(sample_img_path)
                        st.image(sample_image, caption=f'{predicted_class_name} 예시 이미지', use_column_width=True)
            except Exception as e:
                st.warning("예시 이미지를 불러올 수 없습니다.")
