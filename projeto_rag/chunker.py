from typing import List


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser maior que zero.")
    if overlap < 0:
        raise ValueError("overlap nao pode ser negativo.")
    if overlap >= chunk_size:
        raise ValueError("overlap deve ser menor que chunk_size.")

    normalized = text.strip()
    if not normalized:
        return []

    chunks: List[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(normalized):
        chunk = normalized[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks
