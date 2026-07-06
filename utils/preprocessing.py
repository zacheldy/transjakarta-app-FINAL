import re
import html
import pandas as pd
import emoji


def basic_clean_text(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = html.unescape(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\bRT\b", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


SLANG_DICT = {
    "ga": "tidak", "gak": "tidak", "gk": "tidak", "ngga": "tidak", "nggak": "tidak",
    "ngk": "tidak", "tdk": "tidak", "tak": "tidak", "bkn": "bukan",
    "yg": "yang", "dgn": "dengan", "dg": "dengan", "dr": "dari", "dri": "dari",
    "krn": "karena", "karna": "karena", "klo": "kalau", "kl": "kalau", "kalo": "kalau",
    "tp": "tapi", "tapii": "tapi", "sm": "sama", "sma": "sama", "aja": "saja", "aj": "saja",
    "bgt": "banget", "bgtt": "banget", "bangett": "banget",
    "udh": "sudah", "udah": "sudah", "sdh": "sudah", "blm": "belum", "belom": "belum",
    "brp": "berapa", "hrs": "harus", "kyk": "seperti", "kayak": "seperti", "kek": "seperti",
    "emg": "memang", "emang": "memang", "lg": "lagi", "lgi": "lagi", "org": "orang",
    "pd": "pada", "utk": "untuk", "dl": "dulu", "dlu": "dulu", "bs": "bisa", "bsa": "bisa",
    "td": "tadi", "kmrn": "kemarin", "skrg": "sekarang", "skrng": "sekarang",
    "smpe": "sampai", "sampe": "sampai", "ampe": "sampai",
    "nunggu": "menunggu", "nuggu": "menunggu", "ntr": "nanti",
    "min": "admin", "mins": "admin", "mimin": "admin", "gan": "admin", "kak": "kakak",
    "thx": "terima kasih", "thanks": "terima kasih", "makasih": "terima kasih",
    "makasi": "terima kasih", "mksh": "terima kasih",
    "wkwk": "tertawa", "wkwkwk": "tertawa", "wkwwk": "tertawa", "haha": "tertawa", "hehe": "tertawa",
}

DOMAIN_DICT = {
    "tj": "transjakarta", "transjkt": "transjakarta", "transjakarta": "transjakarta",
    "trans jakarta": "transjakarta", "busway": "transjakarta",
    "jak lingko": "jaklingko", "jak-lingko": "jaklingko",
    "tapin": "tap in", "tapout": "tap out", "tapping": "tap",
    "e-money": "emoney", "e money": "emoney",
}


def normalize_repeated_chars(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def replace_phrases(text: str, phrase_dict: dict) -> str:
    for k, v in sorted(phrase_dict.items(), key=lambda x: len(x[0]), reverse=True):
        if " " in k or "-" in k:
            text = re.sub(rf"\b{re.escape(k)}\b", v, text)
    return text


def normalize_tokens(text: str) -> str:
    text = text.lower()
    text = normalize_repeated_chars(text)
    text = replace_phrases(text, DOMAIN_DICT)

    tokens = text.split()
    normalized = []
    for tok in tokens:
        clean_tok = tok.strip(".,!?;:'\"()[]{}")
        if clean_tok in DOMAIN_DICT:
            clean_tok = DOMAIN_DICT[clean_tok]
        clean_tok = SLANG_DICT.get(clean_tok, clean_tok)
        normalized.append(clean_tok)

    text = " ".join(normalized)
    text = re.sub(r"\s+", " ", text).strip()
    return text


EMOJI_SENTIMENT_MAP = {
    "😭": " sedih ", "😢": " sedih ",
    "😡": " marah ", "😠": " marah ", "🤬": " marah ",
    "😂": " tertawa ", "🤣": " tertawa ",
    "😊": " senang ", "😁": " senang ",
    "😍": " suka ", "❤️": " suka ", "❤": " suka ",
    "👍": " bagus ", "🙏": " terima kasih ",
    "😒": " kesal ", "😤": " kesal ",
    "😔": " kecewa ", "😞": " kecewa ",
}


def convert_emoji_to_words(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    for emo, word in EMOJI_SENTIMENT_MAP.items():
        text = text.replace(emo, word)
    text = emoji.replace_emoji(text, replace=" ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_for_sentiment(text_raw: str) -> str:
    text = basic_clean_text(text_raw)
    text = convert_emoji_to_words(text)
    text = normalize_tokens(text)
    return text
