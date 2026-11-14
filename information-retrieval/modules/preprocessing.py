
import json
import re
from pathlib import Path

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


# Inisialisasi Sastrawi
stemmer = StemmerFactory().create_stemmer()
stopwords = set(StopWordRemoverFactory().get_stop_words())


# Stopword tambahan khusus dataset resep
CUSTOM_STOPWORDS = {
    "gram", "gr", "kg", "mg", "ml",
    "liter", "sdt", "sdm", "cup",
    "cups", "cangkir", "sachet",
    "bungkus", "sejumput", "secukupnya",
    "opsional", "optional"
}


# 1. Case Folding
def case_folding(text):
    return text.lower()


# 2. Punctuation Removal
def punctuation_removal(text):
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# 3. Tokenization
def tokenization(text):
    return text.split()


# 4. Stopword Removal
def stopword_removal(tokens):
    return [
        word for word in tokens
        if word not in stopwords
        and word not in CUSTOM_STOPWORDS
    ]


# 5. Stemming
def stemming(tokens):
    return [stemmer.stem(word) for word in tokens]


# Pipeline preprocessing untuk query maupun teks
def preprocess_text(text):
    # Case Folding
    text = case_folding(text)

    # Punctuation Removal
    text = punctuation_removal(text)

    # Tokenization
    tokens = tokenization(text)

    # Stopword Removal sebelum stemming
    tokens = stopword_removal(tokens)

    # Stemming menggunakan Sastrawi
    tokens = stemming(tokens)

    # Stopword Removal kembali setelah stemming
    tokens = stopword_removal(tokens)

    # Menghilangkan token duplikat
    tokens = list(dict.fromkeys(tokens))

    return tokens


# Preprocessing satu dokumen resep
def preprocess_document(doc):
    judul = doc.get("judul", "")

    bahan = doc.get("bahan", [])
    if isinstance(bahan, list):
        bahan_text = " ".join(str(item) for item in bahan)
    else:
        bahan_text = str(bahan)

    # Menggabungkan judul dan bahan
    combined_text = f"{judul} {bahan_text}"

    # Menghasilkan token unik
    tokens = preprocess_text(combined_text)

    # Mempertahankan seluruh informasi resep
    processed_doc = doc.copy()
    processed_doc["tokens"] = tokens

    return processed_doc


# Preprocessing seluruh dataset
def preprocess_dataset(dataset):
    return [
        preprocess_document(doc)
        for doc in dataset
    ]


# Preprocessing query untuk pencarian Streamlit
def preprocess_query(query):
    return preprocess_text(query)


# Menjalankan preprocessing dataset
def main():
    base_dir = Path(__file__).resolve().parent.parent

    input_file = base_dir / "dataset_pangan.json"
    output_file = base_dir / "preprocessed_dataset.json"

    with open(input_file, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError("Dataset harus berbentuk list JSON.")

    processed_dataset = preprocess_dataset(dataset)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(
            processed_dataset,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("Preprocessing berhasil!")
    print(f"Jumlah resep: {len(processed_dataset)}")
    print(f"File output: {output_file}")


if __name__ == "__main__":
    main()
