def calculate_jaccard(query_tokens, doc_tokens):
    """Skor Jaccard Similarity antara token query dan token dokumen. Rumus: |Q ∩ D| / |Q ∪ D|
    """
    set_q = set(query_tokens)
    set_d = set(doc_tokens)

    irisan = set_q.intersection(set_d)
    gabungan = set_q.union(set_d)

    if not gabungan:
        return 0.0

    return len(irisan) / len(gabungan)


def calculate_dsc(query_tokens, doc_tokens):
    """Skor Sørensen-Dice Coefficient (DSC) antara token query dan token dokumen. Rumus: 2 * |Q ∩ D| / (|Q| + |D|)
    """
    set_q = set(query_tokens)
    set_d = set(doc_tokens)

    irisan = set_q.intersection(set_d)
    total_elemen = len(set_q) + len(set_d)

    if total_elemen == 0:
        return 0.0

    return (2 * len(irisan)) / total_elemen


def rank_documents(query_tokens, dataset_preprocessed, method="jaccard"):
    """Skor untuk seluruh dokumen dalam dataset, lalu mengurutkan (ranking) dari skor tertinggi ke terendah. Membuang dokumen dengan skor 0.0.
    """
    results = []

    for doc in dataset_preprocessed:
        doc_tokens = doc.get("tokens", [])

        if method == "jaccard":
            score = calculate_jaccard(query_tokens, doc_tokens)
        elif method == "dsc":
            score = calculate_dsc(query_tokens, doc_tokens)
        else:
            score = 0.0

        if score > 0.0:
            doc_result = doc.copy()
            doc_result["score"] = round(score, 4)
            results.append(doc_result)

    #urutan dsc
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    return results_sorted
