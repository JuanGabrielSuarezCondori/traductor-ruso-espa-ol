import os
import re
import io
import zipfile
import urllib.request
from typing import List, Tuple

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "ru_es.tsv")
TATOEBA_URL = "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/es-ru.txt.zip"

def clean_text_ru(text: str) -> str:
    """Limpieza básica para oraciones en ruso."""
    text = text.lower().strip()
    text = re.sub(r"[^а-яё0-9\s]", " ", text)
    tokens = text.split()
    return " ".join(tokens)

def clean_text_es(text: str) -> str:
    """Limpieza básica para oraciones en español."""
    text = text.lower().strip()
    text = re.sub(r"[^a-záéíóúüñ0-9\s]", " ", text)
    tokens = text.split()
    return " ".join(tokens)

def get_core_vocabulary_pairs() -> List[Tuple[str, str]]:
    """
    Diccionario esencial de vocabulario y frases cotidianas RU-ES.
    Incluye palabras individuales críticas (saludos, pronombres, sustantivos,
    adjetivos, adverbios y verbos básicos) para garantizar que el modelo
    aprenda traducciones directas de 1 palabra y frases cortas.
    """
    pairs = [
        # Saludos y cortesía
        ("привет", "hola"),
        ("здравствуйте", "hola"),
        ("доброе утро", "buenos días"),
        ("добрый день", "buenas tardes"),
        ("добрый вечер", "buenas noches"),
        ("спокойной ночи", "buenas noches"),
        ("пока", "adiós"),
        ("до свидания", "adiós"),
        ("спасибо", "gracias"),
        ("большое спасибо", "muchas gracias"),
        ("пожалуйста", "por favor"),
        ("не за что", "de nada"),
        ("извините", "perdón"),
        ("прости", "lo siento"),
        ("да", "sí"),
        ("нет", "no"),
        ("ладно", "de acuerdo"),
        ("конечно", "por supuesto"),
        ("как дела", "cómo estás"),
        ("как поживаешь", "cómo te va"),
        ("как тебя зовут", "cómo te llamas"),
        ("меня зовут", "me llamo"),
        ("приятно познакомиться", "mucho gusto"),

        # Preguntas clave
        ("что", "qué"),
        ("кто", "quién"),
        ("где", "dónde"),
        ("куда", "adónde"),
        ("откуда", "de dónde"),
        ("когда", "cuándo"),
        ("почему", "por qué"),
        ("как", "cómo"),
        ("сколько", "cuánto"),

        # Pronombres personales
        ("я", "yo"),
        ("ты", "tú"),
        ("он", "él"),
        ("она", "ella"),
        ("оно", "eso"),
        ("мы", "nosotros"),
        ("вы", "ustedes"),
        ("они", "ellos"),

        # Sustantivos cotidianos
        ("дом", "casa"),
        ("книга", "libro"),
        ("вода", "agua"),
        ("хлеб", "pan"),
        ("чай", "té"),
        ("кофе", "café"),
        ("молоко", "leche"),
        ("яблоко", "manzana"),
        ("солнце", "sol"),
        ("луна", "luna"),
        ("небо", "cielo"),
        ("море", "mar"),
        ("город", "ciudad"),
        ("страна", "país"),
        ("мир", "mundo"),
        ("человек", "persona"),
        ("люди", "personas"),
        ("мужчина", "hombre"),
        ("женщина", "mujer"),
        ("ребенок", "niño"),
        ("дети", "niños"),
        ("друг", "amigo"),
        ("подруга", "amiga"),
        ("мама", "mamá"),
        ("папа", "papá"),
        ("семья", "familia"),
        ("брат", "hermano"),
        ("сестра", "hermana"),
        ("сын", "hijo"),
        ("дочь", "hija"),
        ("кот", "gato"),
        ("кошка", "gato"),
        ("собака", "perro"),
        ("машина", "coche"),
        ("поезд", "tren"),
        ("время", "tiempo"),
        ("день", "día"),
        ("ночь", "noche"),
        ("утро", "mañana"),
        ("вечер", "tarde"),
        ("работа", "trabajo"),
        ("школа", "escuela"),
        ("деньги", "dinero"),
        ("музыка", "música"),
        ("жизнь", "vida"),

        # Adjetivos y adverbios
        ("хорошо", "bien"),
        ("плохо", "mal"),
        ("хороший", "bueno"),
        ("плохой", "malo"),
        ("большой", "grande"),
        ("маленький", "pequeño"),
        ("новый", "nuevo"),
        ("старый", "viejo"),
        ("красивый", "bonito"),
        ("быстро", "rápido"),
        ("медленно", "despacio"),
        ("очень", "muy"),
        ("много", "mucho"),
        ("мало", "poco"),
        ("здесь", "aquí"),
        ("там", "allí"),
        ("сейчас", "ahora"),
        ("сегодня", "hoy"),
        ("вчера", "ayer"),
        ("завтра", "mañana"),
        ("всегда", "siempre"),
        ("никогда", "nunca"),

        # Frases y verbos esenciales
        ("я знаю", "lo sé"),
        ("я не знаю", "no lo sé"),
        ("я понимаю", "entiendo"),
        ("я не понимаю", "no entiendo"),
        ("я говорю по русски", "hablo ruso"),
        ("я говорю по испански", "hablo español"),
        ("я люблю тебя", "te amo"),
        ("я хочу", "yo quiero"),
        ("я хочу спать", "tengo sueño"),
        ("мне пора идти спать", "tengo que irme a dormir"),
        ("мне пора", "tengo que irme"),
        ("что ты делаешь", "qué estás haciendo"),
        ("где ты живешь", "dónde vives"),
        ("где ты", "dónde estás"),
        ("кто ты", "quién eres"),
        ("что это", "qué es esto"),
        ("это мой дом", "esta es mi casa"),
        ("это хорошо", "esto está bien"),
        ("это правда", "es verdad"),
        ("я студент", "soy estudiante"),
        ("он мой друг", "él es mi amigo"),
        ("она моя подруга", "ella es mi amiga"),
        ("сегодня хороший день", "hoy es un buen día"),
        ("давайте попробуем", "vamos a intentar")
    ]
    return pairs

def download_and_extract_tatoeba(target_pairs: int = 12000, min_len: int = 1, max_len: int = 6) -> List[Tuple[str, str]]:
    """
    Descarga Tatoeba RU-ES desde OPUS y selecciona pares de longitud 1 a 6.
    Garantiza la inclusión de pares de 1 sola palabra y oraciones breves directas.
    """
    print(f"Descargando corpus Tatoeba desde {TATOEBA_URL}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(TATOEBA_URL, headers=headers)
    
    with urllib.request.urlopen(req, timeout=60) as response:
        content = response.read()
        
    print(f"Archivo descargado ({len(content) / (1024 * 1024):.2f} MB). Extrayendo pares...")
    
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        es_raw = z.open("Tatoeba.es-ru.es").read().decode("utf-8").splitlines()
        ru_raw = z.open("Tatoeba.es-ru.ru").read().decode("utf-8").splitlines()
        
    print(f"Total pares crudos: {len(ru_raw)}")
    
    # 1. Recolectar todos los pares de 1 palabra primero
    single_word_pairs = []
    short_pairs = []
    seen = set()

    # Añadir pares de vocabulario esencial reforzados (multiplicados para dar peso)
    core_pairs = get_core_vocabulary_pairs()
    boosted_core = []
    for _ in range(12):  # Repetir para garantizar que las palabras base tengan fuerte señal
        boosted_core.extend(core_pairs)

    for ru_line, es_line in zip(ru_raw, es_raw):
        ru_clean = clean_text_ru(ru_line)
        es_clean = clean_text_es(es_line)
        
        ru_words = ru_clean.split()
        es_words = es_clean.split()
        
        if 1 <= len(ru_words) <= max_len and 1 <= len(es_words) <= max_len:
            key = (ru_clean, es_clean)
            if key not in seen:
                seen.add(key)
                if len(ru_words) == 1:
                    single_word_pairs.append((ru_clean, es_clean))
                else:
                    short_pairs.append((ru_clean, es_clean))

    print(f"Pares de 1 palabra en Tatoeba: {len(single_word_pairs)}")
    print(f"Pares cortos (2-6 palabras) en Tatoeba: {len(short_pairs)}")

    # Combinar: Core reforzado + Pares de 1 palabra de Tatoeba + oraciones cortas hasta target_pairs
    combined = []
    combined.extend(boosted_core)
    combined.extend(single_word_pairs)
    
    remaining = target_pairs - len(combined)
    if remaining > 0:
        combined.extend(short_pairs[:remaining])
        
    return combined

def prepare_dataset(num_samples: int = 12000):
    """Prepara el dataset enriquecido y lo guarda en formato TSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    try:
        pairs = download_and_extract_tatoeba(target_pairs=num_samples, min_len=1, max_len=6)
    except Exception as e:
        print(f"Aviso: Error al descargar ({e}). Usando vocabulario esencial offline...")
        core = get_core_vocabulary_pairs()
        pairs = []
        for _ in range(30):
            pairs.extend(core)
            
    print(f"Guardando {len(pairs)} pares RU-ES en {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ru, es in pairs:
            f.write(f"{ru}\t{es}\n")
            
    print("Dataset preparado con éxito.")
    # Contar oraciones de 1 palabra
    single_count = sum(1 for r, _ in pairs if len(r.split()) == 1)
    print(f"Total pares: {len(pairs)} | Pares de 1 sola palabra: {single_count}")
    print("Muestras de ejemplo:")
    for ru, es in pairs[:10]:
        print(f"  RU: {ru:25} -> ES: {es}")

if __name__ == "__main__":
    prepare_dataset(num_samples=12000)
