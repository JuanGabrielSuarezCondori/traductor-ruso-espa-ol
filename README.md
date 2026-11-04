# Traductor Neuronal Ruso ➔ Español (RNN Many-to-Many Asíncrono)

Implementación didáctica y minimalista de un sistema de traducción automática de texto a texto de **Ruso (RU)** a **Español (ES)** utilizando una red neuronal recurrente basada en el esquema clásico **many-to-many asíncrono / desacoplado** (Encoder-Decoder elemental de Andrej Karpathy), acompañado de una interfaz visual interactiva en **Streamlit**.

---

## 🧠 Arquitectura del Modelo

El modelo sigue estrictamente el patrón clásico **Many-to-Many Desacoplado**:

```text
       [x₁]    [x₂]    ...    [x_n]               <-- FASE 1: Lectura (Encoder GRU)
        ↓       ↓              ↓
      [h₁] -> [h₂] -> ... -> [h_n]   ═════════╗ (Traspaso del vector de contexto)
                                             ║
                                             ▼
                                           [s₀] -> [s₁]   ->   [s₂]   -> ... -> [s_m]
                                            ↓       ↓           ↓                 ↓
                                          <sos>    [y₁]        [y₂]             <eos>
                                                  (hola)      (amigo)
                                                  <-- FASE 2: Generación (Decoder GRU)
```

1. **Fase 1 (Lectura / Entradas $x$):** Una celda recurrente (GRU) procesa secuencialmente cada token de la frase en ruso ($x_1, \dots, x_n$) actualizando su estado oculto ($h_1, \dots, h_n$) sin generar predicciones intermedias.
2. **Traspaso:** El último estado oculto del encoder ($h_n$) resume toda la oración y se transfiere como estado oculto inicial del decoder ($s_0 = h_n$).
3. **Fase 2 (Generación / Salidas $y$):** A partir de $s_0$ y el token inicial `<sos>`, el decoder emite paso a paso los tokens en español ($y_1, \dots, y_m$) retroalimentando el token predicho al siguiente paso hasta encontrar el token de parada `<eos>` o un límite máximo de pasos.
4. **Sin mecanismos de atención ni Transformers:** Arquitectura pura para reflejar fielmente la base conceptual de las secuencias recurrentes.

---

## 📁 Estructura del Proyecto

```text
TRADUCTOR RNN/
├── data_prep.py      # Descarga y filtrado de pares paralelos RU-ES (Tatoeba / OPUS)
├── train.py          # Arquitectura Many-to-Many (PyTorch), loop de entrenamiento y guardado
├── app.py            # Interfaz web en Streamlit con visualización paso a paso
├── requirements.txt  # Dependencias (torch, streamlit, requests)
├── model.pt          # Checkpoint del modelo entrenado y vocabularios
└── data/
    └── ru_es.tsv     # Corpus paralelo RU-ES (8,000 pares limpios y didácticos)
```

---

## 🚀 Instalación y Uso

### 1. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 2. Preparar el Dataset (Opcional si ya existe data/ru_es.tsv)
```bash
python data_prep.py
```

### 3. Entrenar el Modelo (Opcional si ya tienes model.pt)
```bash
python train.py
```

### 4. Ejecutar la Aplicación Interactiva en Streamlit
```bash
streamlit run app.py
```
Abre en tu navegador `http://localhost:8501`.
