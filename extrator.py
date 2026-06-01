from projeto_rag.document_loader import extract_text


def extrair_texto_do_pdf(caminho_do_pdf):
    return extract_text(caminho_do_pdf)


if __name__ == "__main__":
    print("Use: python -m projeto_rag.cli index --path ManualCafe.pdf")
