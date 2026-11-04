import os
import sys
import torch
import streamlit as st
import pandas as pd
from typing import Tuple, List, Dict

# Importar arquitectura y utilidades desde train.py
from train import (
    Vocab,
    Encoder,
    Decoder,
    Seq2Seq,
    translate_sentence,
    PAD_IDX,
    SOS_IDX,
    EOS_IDX,
    UNK_IDX
)

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Traductor RU ➔ ES | RNN Many-to-Many",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para una interfaz moderna, limpia y didáctica
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 58, 138, 0.3) 100%);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }
    .result-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #38bdf8;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .result-text {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.3;
    }
    .step-badge {
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }
    .step-badge-eos {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_translator_model(checkpoint_path: str = "model.pt"):
    """
    Carga el modelo Seq2Seq y los vocabularios en memoria usando st.cache_resource.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(checkpoint_path):
        alt_path = os.path.join("checkpoints", checkpoint_path)
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            raise FileNotFoundError(f"No se encontró el checkpoint '{checkpoint_path}'. Ejecuta train.py primero.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    src_vocab = Vocab()
    src_vocab.word2idx = checkpoint["src_vocab_word2idx"]
    src_vocab.idx2word = checkpoint["src_vocab_idx2word"]

    trg_vocab = Vocab()
    trg_vocab.word2idx = checkpoint["trg_vocab_word2idx"]
    trg_vocab.idx2word = checkpoint["trg_vocab_idx2word"]

    emb_dim = checkpoint.get("emb_dim", 128)
    hidden_dim = checkpoint.get("hidden_dim", 256)

    encoder = Encoder(input_dim=len(src_vocab), emb_dim=emb_dim, hidden_dim=hidden_dim).to(device)
    decoder = Decoder(output_dim=len(trg_vocab), emb_dim=emb_dim, hidden_dim=hidden_dim).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    metadata = {
        "device": str(device),
        "src_vocab_size": len(src_vocab),
        "trg_vocab_size": len(trg_vocab),
        "emb_dim": emb_dim,
        "hidden_dim": hidden_dim,
        "epoch": checkpoint.get("epoch", "-"),
        "val_loss": checkpoint.get("val_loss", None)
    }

    return model, src_vocab, trg_vocab, metadata


# ==============================================================================
# BARRA LATERAL (INFORMACIÓN DEL MODELO Y METADATOS)
# ==============================================================================

with st.sidebar:
    st.header("⚙️ Modelo Recurrente")

    if st.button("🔄 Recargar Modelo en Memoria", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    try:
        model, src_vocab, trg_vocab, meta = load_translator_model("model.pt")
        st.success("✅ Modelo cargado exitosamente")

        st.markdown("### Parámetros de la Red")
        st.markdown(f"""
        - **Patrón:** Many-to-Many Asíncrono
        - **Celda RNN:** GRU con `pack_padded_sequence`
        - **Dim. Oculta ($h$):** `{meta['hidden_dim']}`
        - **Dim. Embedding:** `{meta['emb_dim']}`
        - **Vocabulario RU:** `{meta['src_vocab_size']}` tokens
        - **Vocabulario ES:** `{meta['trg_vocab_size']}` tokens
        - **Época Entrenada:** `{meta['epoch']}`
        - **Dispositivo:** `{meta['device'].upper()}`
        """)

        if meta.get("val_loss") is not None:
            st.metric("Pérdida de Validación", f"{meta['val_loss']:.4f}")

    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        st.info("Asegúrate de haber ejecutado `train.py` para generar `model.pt`.")
        model = None

    st.markdown("---")
    max_decode_len = st.slider("Límite de pasos del decoder:", min_value=3, max_value=25, value=15)
    st.caption("Arquitectura pura: Sin atención ni Transformers (esquema elemental de Karpathy).")


# ==============================================================================
# CUERPO PRINCIPAL
# ==============================================================================

st.markdown('<div class="main-title">Traductor Ruso ➔ Español</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Arquitectura Many-to-Many Asíncrona (Encoder-Decoder Clásico de Karpathy)</div>', unsafe_allow_html=True)

with st.expander("📘 ¿Cómo funciona este esquema Many-to-Many Asíncrono?", expanded=False):
    st.markdown("""
    En el diagrama clásico de **Andrej Karpathy** (*The Unreasonable Effectiveness of Recurrent Neural Networks*), 
    el patrón **many-to-many asíncrono** (desacoplado) resuelve secuencias donde la entrada $[x_1 \dots x_T]$ y la salida $[y_1 \dots y_{T'}]$ tienen longitudes arbitrarias:

    ```text
       [x1]    [x2]    ...    [xn]                <-- Fase 1: Lectura (Encoder GRU)
        ↓       ↓              ↓
      [h1] -> [h2] -> ... -> [hn]   ═════════╗ (Traspaso del vector de contexto)
                                             ║
                                             ▼
                                           [s0] -> [s1]   ->   [s2]   -> ... -> [sm]
                                            ↓       ↓           ↓                 ↓
                                          <sos>    [y1]        [y2]             <eos>
                                                  (hola)      (amigo)
                                                  <-- Fase 2: Generación (Decoder GRU)
    ```

    1. **Fase 1 (Encoder):** Procesa cada palabra rusa paso a paso. Gracias a `pack_padded_sequence`, el estado oculto final $h_n$ corresponde exactamente al último token real de la frase (sin corromperse con `<pad>`).
    2. **Traspaso:** El último estado oculto $h_n$ se asigna directamente como estado inicial $s_0$ del Decoder.
    3. **Fase 2 (Decoder):** Emite paso a paso los tokens en español $y_1, y_2, \dots$ retroalimentando cada token al paso siguiente hasta emitir `<eos>`.
    """)

# Ejemplos rápidos: palabras solas y frases cortas
st.markdown("##### 💡 Prueba rápida (Palabras solas y frases cortas):")
sample_words = [
    "привет",
    "спасибо",
    "кот",
    "хорошо",
    "да",
    "нет",
    "что ты делаешь",
    "где ты",
    "я говорю по русски",
    "мне пора идти спать"
]

row1_cols = st.columns(5)
for i in range(5):
    if row1_cols[i].button(sample_words[i], key=f"btn_w_{i}", use_container_width=True):
        st.session_state.input_ru_text = sample_words[i]

row2_cols = st.columns(5)
for i in range(5, 10):
    if row2_cols[i - 5].button(sample_words[i], key=f"btn_w_{i}", use_container_width=True):
        st.session_state.input_ru_text = sample_words[i]

if "input_ru_text" not in st.session_state:
    st.session_state.input_ru_text = "привет"

ru_input = st.text_input(
    "Ingresa una palabra o frase en ruso:",
    value=st.session_state.input_ru_text,
    key="ru_input_text_field",
    placeholder="Ej: привет, спасибо, кот, что ты делаешь..."
)

translate_btn = st.button("🚀 Traducir al Español", type="primary", use_container_width=True)

if translate_btn or ru_input:
    if not ru_input.strip():
        st.warning("Por favor ingresa una palabra o frase en ruso.")
    elif model is None:
        st.error("El modelo no está disponible. Entrénalo primero ejecutando `python train.py`.")
    else:
        with st.spinner("Procesando en el Encoder y generando en el Decoder..."):
            translation, step_details = translate_sentence(
                model=model,
                ru_sentence=ru_input,
                src_vocab=src_vocab,
                trg_vocab=trg_vocab,
                max_len=max_decode_len,
                device=torch.device(meta["device"])
            )

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">🇪🇸 Traducción al Español (Salida Decoder)</div>
            <div class="result-text">{translation if translation else '(sin salida)'}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("### 🔍 Traza de Generación Paso a Paso (Fase 2: Decoder)")
        st.caption("Muestra cada token generado en cada paso temporal $t$ con su probabilidad softmax.")

        if step_details:
            chips_html = '<div style="margin-bottom: 1rem;">'
            for item in step_details:
                w = item["token_word"]
                is_eos = (w == "<eos>")
                css_cls = "step-badge-eos" if is_eos else "step-badge"
                chips_html += f'<span class="step-badge {css_cls}">Paso {item["step"]}: <b>{w}</b> ({item["probability"]*100:.1f}%)</span>'
            chips_html += '</div>'
            st.markdown(chips_html, unsafe_allow_html=True)

            table_data = []
            for item in step_details:
                table_data.append({
                    "Paso (t)": f"t = {item['step']}",
                    "Token Generado (y_t)": item["token_word"],
                    "ID Vocabulario": item["token_id"],
                    "Confianza Softmax": f"{item['probability'] * 100:.2f}%"
                })

            df_steps = pd.DataFrame(table_data)
            st.dataframe(
                df_steps,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No se generaron tokens. Verifica que la entrada no esté vacía.")

st.markdown("---")
st.caption("Implementado con PyTorch & Streamlit | Red Neuronal Recurrente Many-to-Many Desacoplada sin Atención")
