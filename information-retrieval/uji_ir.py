
import json

from modules.preprocessing import preprocess_query
from modules.ir_engine import rank_documents


# 1. Membaca dataset hasil preprocessing
with open("preprocessed_dataset.json", "r", encoding="utf-8") as file:
    dataset = json.load(file)

print("=" * 50)
print("PENGUJIAN INFORMATION RETRIEVAL")
print("=" * 50)

# 2. Memasukkan kata kunci pencarian
query = "nasi goreng porang"

# 3. Melakukan preprocessing pada query
query_tokens = preprocess_query(query)

print("\nKata kunci:", query)
print("Query tokens:", query_tokens)

# 4. Menguji metode Jaccard
hasil_jaccard = rank_documents(
    query_tokens,
    dataset,
    method="jaccard"
)

# 5. Menguji metode Sørensen-Dice
hasil_dsc = rank_documents(
    query_tokens,
    dataset,
    method="dsc"
)


# 6. Menampilkan hasil Jaccard
print("\nHASIL JACCARD SIMILARITY")
print("-" * 50)

for i, doc in enumerate(hasil_jaccard[:5], start=1):
    print(f"{i}. {doc['judul']}")
    print(f"   Skor: {doc['score']}")


# 7. Menampilkan hasil Sørensen-Dice
print("\nHASIL SORENSEN-DICE")
print("-" * 50)

for i, doc in enumerate(hasil_dsc[:5], start=1):
    print(f"{i}. {doc['judul']}")
    print(f"   Skor: {doc['score']}")

print("\nPengujian selesai!")
