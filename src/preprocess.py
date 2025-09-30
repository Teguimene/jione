################ Functions to extract answer #########
import re
# For truthfulqa
def detect_choice(text: str, choices: list[str]) -> str | None:
    text_clean = text.lower().strip()

    for choice in choices:
        if text_clean == choice.lower():
            return choice

    for choice in choices:
        if choice.lower() in text_clean:
            return choice

    return "None"

# For mbpp
def extract_before_system(text: str) -> str:
    """
    1) S'il existe un bloc :
       Corrected Code:
       ```python
       ...code...
       ```
       → renvoie uniquement le code (sans les backticks).
    2) Sinon : extrait tout le contenu avant 'system' et supprime 'system' + la suite.
    """
    # 1) Tenter d'extraire le bloc Corrected Code
    m = re.search(r"Corrected Code:\s*```(?:python)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2) Comportement original
    index = text.find("system")
    if index != -1:
        return text[:index].strip()
    return text.strip()

# For sst-2
def extract_sentiment(text):
    # print(text)
    # Nettoyer le texte et chercher POSITIVE ou NEGATIVE en début de ligne
    match = re.search(r'^(?:\W*([Pp][Oo][Ss][Ii][Tt][Ii][Vv][Ee])|\W*([Nn][Ee][Gg][Aa][Tt][Ii][Vv][Ee]))', text.strip())
    if match:
        if match.group(1):  # POSITIVE trouvé
            return 'positive'
        elif match.group(2):  # NEGATIVE trouvé
            return 'negative'
    return 'neutral'  # Aucun des deux trouvé

# For gsm8k
def extract_final_answer(prediction: str) -> str:
    prediction = prediction.strip()

    match = re.search(r'(?:####\s*([\d,]+(?:\.\d+)?)|([\d,]+(?:\.\d+)?)\s*####)', prediction)
    if match:
        return (match.group(1) or match.group(2)).strip()

    lines = prediction.splitlines()
    for line in lines:
        line = line.strip()
        if re.fullmatch(r'[\d,]+(?:\.\d+)?', line):
            return line

    matches = re.findall(r'=\s*\$?\s*([\d,]+(?:\.\d+)?)', prediction)
    if matches:
        return matches[-1].strip()

    matches = re.findall(r'\$\s*([\d,]+(?:\.\d+)?)', prediction)
    if matches:
        return matches[-1].strip()

    matches = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b', prediction)
    if matches:
        return matches[-1].strip()

    return prediction
