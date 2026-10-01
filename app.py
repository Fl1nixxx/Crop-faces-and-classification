import hashlib
from PIL import Image
import streamlit as st
from streamlit_cropper import st_cropper
from src.model import load_model
from src.face_crop import (create_detector,mtcnn_crop,click_crop,find_face_in_click_crop)
from streamlit_image_coordinates import streamlit_image_coordinates
from src.inference import (classify_faces,filter_results,AGE_LABELS,RACE_LABELS,GENDER_LABELS)

st.set_page_config(page_title="Face Filter",page_icon="🧠",layout="wide")

@st.cache_resource
def get_model():
    return load_model("filternet_weights.pth")

@st.cache_resource
def get_detector():
    return create_detector()

model, device = get_model()
detector = get_detector()

if "manual_faces" not in st.session_state:
    st.session_state["manual_faces"] = []

if "mtcnn_faces" not in st.session_state:
    st.session_state["mtcnn_faces"] = []

if "results" not in st.session_state:
    st.session_state["results"] = []

if "current_image" not in st.session_state:
    st.session_state["current_image"] = None

if "last_click" not in st.session_state:
    st.session_state["last_click"] = None

if "clicked_faces" not in st.session_state:
    st.session_state["clicked_faces"] = []

st.title("Face Crop + Classification")

with st.sidebar:
    st.header("Фильтр")

    age_filter = st.multiselect("Возраст",AGE_LABELS,placeholder="Любой")
    race_filter = st.multiselect("Race",RACE_LABELS,placeholder="Любой")
    gender_filter = st.multiselect("Gender",GENDER_LABELS,placeholder="Любой")

    show_only_filtered = st.checkbox("Показывать только подходящие",value=True)

uploaded = st.file_uploader("Загрузите фотографию",type=["jpg","jpeg","png","webp"])


if uploaded is None:
    st.info("Сначала загрузите фотографию")
    st.stop()

image_bytes = uploaded.getvalue()
image_hash = hashlib.md5(image_bytes).hexdigest()


if (st.session_state["current_image"]!= image_hash):

    st.session_state["current_image"] = image_hash
    st.session_state["manual_faces"] = []
    st.session_state["mtcnn_faces"] = []
    st.session_state["results"] = []
    st.session_state["last_click"] = None


image = Image.open(uploaded).convert("RGB")
st.write(f"Размер исходника: "f"{image.width} × "f"{image.height}")

mode = st.radio("Способ выделения лиц",["MTCNN","Вручную","Клик по лицу"],horizontal=True)

#Первый режим
if mode == "MTCNN":
    st.subheader("Автоматический поиск")

    col1, col2 = st.columns([1,2])

    with col1:
        confidence = st.slider("MTCNN confidence",min_value=0.50,max_value=0.999,value=0.95,step=0.001)
        min_face_size = st.slider("Минимальный размер лица",min_value=20,max_value=300,value=40,step=5)
        padding = st.slider("Отступ %",min_value=0, max_value=50,value=15)


        if st.button("Найти лица",type="primary",use_container_width=True):
            faces = mtcnn_crop(image=image,detector=detector,confidence_threshold=confidence,min_face_size=min_face_size,padding=padding / 100)

            st.session_state["mtcnn_faces"] = faces
            st.session_state["results"] = []


    with col2:
        st.image(image,use_container_width=True)

    faces = st.session_state["mtcnn_faces"]

    if faces:
        st.subheader(f"Найдено лиц: "f"{len(faces)}")
      
        cols = st.columns(min(4,len(faces)))


        for i, face in enumerate(faces):
            with cols[i % len(cols)]:

                st.image(face["image"],width=min(250,face[ "image"].width))
                st.caption(f"confidence: "f"{face['confidence']:.3f}")

        if st.button("Распознать все лица",type="primary"):
            st.session_state["results"] = classify_faces(faces,model,device)

#Второй режим 
elif mode == "Вручную":
    st.subheader("Ручное выделение")
  
    crop = st_cropper(image,realtime_update=True,box_color="#ff0000",aspect_ratio=None, key=f"cropper_{image_hash}")
  
    st.write("Текущая выделенная область")
    st.image(crop,width=min(400,crop.width))


    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Добавить лицо",use_container_width=True):
            if (crop.width >= 20 and crop.height >= 20):

                st.session_state["manual_faces"].append({"image":crop.copy(), "source": "Manual","confidence":None})
                st.session_state["results"] = []

    with col2:

        if st.button("Удалить последнее",use_container_width=True):
            if st.session_state["manual_faces"]:
                st.session_state["manual_faces"].pop()
                st.session_state["results"] = []

    with col3:

        if st.button("Очистить",use_container_width=True):
            st.session_state["manual_faces"] = []
            st.session_state["results"] = []


    faces = st.session_state["manual_faces"]

    if faces:
        st.subheader(f"Выбрано лиц: "f"{len(faces)}")

        cols = st.columns(min(4,len(faces)))

        for i, face in enumerate(faces):
            with cols[i % len(cols)]:

                st.image(face["image"],width=min(250,face["image"].width))
                st.caption(f"Лицо №{i + 1}")

        if st.button("Распознать все выбранные",type="primary"):
            st.session_state["results"] = classify_faces(faces,model,device)
            
#Третий режим
elif mode == "Клик по лицу":
    st.subheader("Кликните по лицам")

    max_width = 800

    if image.width > max_width:
        scale = max_width / image.width
        display_image = image.resize((max_width, int(image.height * scale)))
    else:
        display_image = image.copy()

    coordinates = streamlit_image_coordinates(display_image,width=display_image.width,key=f"click_{image_hash}")
    
    if coordinates is not None:
        scale_x = image.width / display_image.width
        scale_y = image.height / display_image.height

        x = int(coordinates["x"] * scale_x)
        y = int(coordinates["y"] * scale_y)

        current_click = (x, y)

        if current_click != st.session_state["last_click"]:
            st.session_state["last_click"] = current_click
            region = click_crop(image=image,x=x,y=y,size=400)

            with st.spinner("Ищу лицо..."):face = find_face_in_click_crop(region,detector,confidence_threshold=0.975)
            if face is None:
                st.warning("В выбранной области лицо не найдено.")
            else:
                st.session_state["clicked_faces"].append(face)
                with st.spinner("Распознаю..."):new_result = classify_faces([face],model,device)
                st.session_state["results"].extend(new_result)

    st.write(f"Распознано лиц: {len(st.session_state['clicked_faces'])}")

    if st.button("Очистить"):
        st.session_state["clicked_faces"] = []
        st.session_state["results"] = []
        st.session_state["last_click"] = None

        st.rerun()

results = st.session_state["results"]

if results:
    filtered = filter_results(results=results, age_filter=age_filter,race_filter=race_filter,gender_filter=gender_filter)
  
    st.divider()

    col1, col2 = st.columns(2)
  
    col1.metric("Всего",len(results))
    col2.metric("Подошло под фильтр",len(filtered))


    if show_only_filtered:
        results_to_show = (filtered)
    else:
        results_to_show = (results)


    if not results_to_show:
        st.warning("Ни одно лицо ""не подходит под фильтр.")

    else:
        st.subheader("Результаты")
        cols = st.columns(3)

        for i, result in enumerate(results_to_show):
            with cols[i % 3]:
              
                st.image(result["image"],width=min( 300,result["image"].width))
                st.markdown(f"### "f"{result['age']}")
                st.write("Race:",result["race"])
                st.write("Gender:",result["gender"])
                st.caption(result["source"])

                if (result.get( "confidence")is not None):
                    st.caption("MTCNN: "f"{result['confidence']:.3f}")
