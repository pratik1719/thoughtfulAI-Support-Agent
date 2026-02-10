from src.retriever import FAQRetriever


def test_retriever_matches_eva():
    qa_items = [
        {"question": "What does EVA do?", "answer": "EVA does eligibility verification."},
        {"question": "What does CAM do?", "answer": "CAM does claims processing."},
    ]
    r = FAQRetriever(qa_items)
    item, score = r.retrieve("Tell me about eligibility verification EVA", threshold=0.10)
    assert item is not None
    assert "EVA" in item["answer"]
    assert score >= 0.10


def test_retriever_returns_none_on_gibberish():
    qa_items = [
        {"question": "What does EVA do?", "answer": "EVA does eligibility verification."},
    ]
    r = FAQRetriever(qa_items)
    item, score = r.retrieve("asdkjasdkljasdklj", threshold=0.50)
    assert item is None
    assert score < 0.50
