import streamlit as st
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Cargamos el modelo FinBERT (entrenado para texto financiero)
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# FinBERT: 0 -> negative, 1 -> neutral, 2 -> positive
LABELS = ["negative", "neutral", "positive"]

# Palabras que queremos EXCLUIR a nivel de query (evitar medallas, etc.)
EXCLUDED_WORDS = ["medal", "olympic", "champion", "award", "album", "movie", "song"]

# Palabras que queremos EXCLUIR en post-filtrado (por si la query no es suficiente)
ADDITIONAL_EXCLUDED = ["dog", "football", "game", "basketball", "wrestling"]

def build_gold_query():
    """
    Construye una query para NewsAPI que busque 'gold' y 'commodity'
    excluyendo las palabras indicadas en EXCLUDED_WORDS.
    """
    base_terms = "(gold OR gold commodity)"
    exclusions = " ".join(f"-{w}" for w in EXCLUDED_WORDS)
    query = f"{base_terms} {exclusions}"
    return query.strip()

def fetch_gold_news(api_key, page_size=20):
    """
    Llama al endpoint /v2/everything de NewsAPI con la query filtrada
    y retorna una lista de artículos.
    """
    query = build_gold_query()

    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&"
        "language=en&"
        f"pageSize={page_size}&"
        f"apiKey={api_key}"
    )
    resp = requests.get(url)
    data = resp.json()

    if data.get("status") == "ok":
        return data.get("articles", [])
    else:
        return []

def post_filter_relevance(articles):
    """
    Filtra las noticias que contengan palabras adicionales excluidas
    (ADDITIONAL_EXCLUDED) en su título o descripción.
    """
    relevant = []
    for art in articles:
        title = (art.get("title") or "").lower()
        desc = (art.get("description") or "").lower()
        combined = f"{title} {desc}"
        # Si contiene alguna de las palabras prohibidas, se descarta
        if any(word in combined for word in ADDITIONAL_EXCLUDED):
            continue
        relevant.append(art)
    return relevant

def analyze_finbert_sentiment(text):
    """
    Aplica FinBERT a un texto, retornando 'negative', 'neutral' o 'positive'.
    """
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)
    label_id = torch.argmax(probs, dim=1).item()
    return LABELS[label_id]

def main():
    st.title("Noticias sobre Oro (gold) con Filtrado y FinBERT")

    # Ingresar API Key de NewsAPI (por defecto, mostramos la solicitada)
    api_key = st.text_input(
        "Ingresa tu NewsAPI Key:",
        value="d8f9ea68afa542f4bad8dcd067d15b01",
        type="password"
    )

    # Elegir cuántas noticias traer
    page_size = st.slider("Número de noticias (1-100)", 1, 100, 20)

    if st.button("Buscar y Analizar"):
        if not api_key:
            st.warning("Por favor, ingresa tu API Key de NewsAPI.")
            return

        # 1) Obtener noticias
        articles = fetch_gold_news(api_key, page_size=page_size)
        st.write(f"Noticias descargadas (antes de filtrar): {len(articles)}")

        # 2) Post-filtrado por palabras irrelevantes
        filtered = post_filter_relevance(articles)
        st.write(f"Noticias relevantes después de filtrar: {len(filtered)}")

        if not filtered:
            st.warning("No quedan noticias relevantes tras el filtrado.")
            return

        # 3) Análisis de sentimiento con FinBERT
        results = []
        for art in filtered:
            title = art.get("title", "")
            description = art.get("description", "")
            content = art.get("content", "")

            full_text = f"{title}. {description}. {content}"
            sentiment = analyze_finbert_sentiment(full_text)

            results.append({
                "title": title,
                "description": description,
                "sentiment": sentiment
            })

        # 4) Mostrar resultados
        st.subheader("Resultados del Análisis")
        for i, res in enumerate(results, start=1):
            st.write(f"**Noticia {i}:**")
            st.write(f"- **Título**: {res['title']}")
            st.write(f"- **Descripción**: {res['description']}")
            st.write(f"- **Sentimiento (FinBERT)**: `{res['sentiment']}`")
            st.write("---")

        # Resumen final
        positives = sum(r["sentiment"] == "positive" for r in results)
        negatives = sum(r["sentiment"] == "negative" for r in results)
        neutrals = sum(r["sentiment"] == "neutral" for r in results)
        total = len(results)

        st.write(f"**Positivas**: {positives} | **Negativas**: {negatives} | **Neutras**: {neutrals} | **Total**: {total}")
        if total > 0:
            positivity_rate = positives / total * 100
            st.metric(
                label="Porcentaje de noticias positivas (FinBERT)",
                value=f"{positivity_rate:.2f}%"
            )

if __name__ == "__main__":
    main()
