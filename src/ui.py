import io
import base64
import streamlit as st
import streamlit.components.v1 as components


def image_actions(image, filename, key):

    buffer = io.BytesIO()
    image.save(buffer,format="PNG")
    image_bytes = buffer.getvalue()
    encoded = base64.b64encode(image_bytes).decode("utf-8")

    col1, col2 = st.columns(2)
  
    with col1:
        st.download_button(label="Скачать",data=image_bytes,file_name=filename,mime="image/png",key=f"download_{key}",width="stretch")
    with col2:
        html = f"""
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: transparent;
            }}

            #copy_{key} {{
                width: 100%;
                height: 40px;

                background-color: #0e1117;
                color: white;

                border: 1px solid #3d444d;
                border-radius: 8px;

                font-size: 16px;
                font-family: sans-serif;

                cursor: pointer;

                transition:
                    background-color 0.15s,
                    border-color 0.15s;
            }}

            #copy_{key}:hover {{
                background-color: #1a1d24;
                border-color: #ff4b4b;
            }}

            #copy_{key}:active {{
                background-color: #262a33;
            }}

        </style>

        <button id="copy_{key}">
            Копировать
        </button>

        <script>
            const button =
                document.getElementById(
                    "copy_{key}"
                );
            button.onclick = async () => {{
                try {{
                    const response = await fetch("data:image/png;base64,{encoded}");
                    const blob = await response.blob();
                    await navigator.clipboard.write([
                        new ClipboardItem({{
                            "image/png": blob
                        }})
                    ]);

                    button.innerText = "Скопировано ✓";
                    
                    setTimeout(() => {{
                        button.innerText ="Копировать";}}, 1500);
                }} catch (error) {{

                    button.innerText = "Не удалось";

                    console.error(error);


                    setTimeout(() => {{
                        button.innerText ="Копировать";}}, 1500);
                }}
            }};
        </script>
        """


        components.html(html,height=55)
