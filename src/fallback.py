import os


def generic_fallback() -> str:
    return (
        "I don’t have a specific canned answer for that yet. "
        "I can help with Thoughtful AI’s agents like EVA (eligibility verification), "
        "CAM (claims processing), and PHIL (payment posting). "
        "Try asking about one of those, or rephrase your question."
    )


def gemini_fallback(user_question: str) -> str:
    """
    Gemini fallback:
    - Requires GOOGLE_API_KEY in env
    - Uses GEMINI_MODEL (default: gemini-2.0-flash)
    - If not available, returns generic fallback
    """
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return generic_fallback()

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        model = genai.GenerativeModel(model_name)

        prompt = (
            "You are a helpful customer support agent for Thoughtful AI.\n"
            "If the user asks something outside the provided FAQ, respond politely and briefly.\n"
            "When relevant, suggest asking about EVA, CAM, PHIL, or benefits of Thoughtful AI’s agents.\n\n"
            f"User question: {user_question}"
        )

        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 200,
            },
        )

        text = getattr(resp, "text", "") or ""
        text = text.strip()
        return text if text else generic_fallback()

    except Exception as e:
        return f"[GEMINI_ERROR] {type(e).__name__}: {e}"
