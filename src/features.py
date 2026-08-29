"""URL 기반 피싱 탐지용 피처 추출 모듈.

발표자료(떡밥팀 최종발표)에서 정의한 15개 피처를 하나의 모듈로 정리했다.
피처는 세 종류로 나뉜다.

1. URL 문자열만으로 계산 (네트워크 요청 불필요) — 빠르고 실패하지 않음
     IP_LIKE, AT, URL_Depth, Redirection, Is_Https, TINY_URL, Check_Hyphen, Query
2. WHOIS 조회 필요 (도메인 등록 정보)
     Domain_Age, Domain_end
3. HTTP 요청 필요 (실제 사이트 접속)
     Mouseover, Web_forwards, Hyperlinks, Domain_Cons
그리고 텍스트 모델(CNN/CountVectorizer)용 토큰열
     Tokenized_url

네트워크가 필요한 피처는 실패 시 -1 을 반환하며(원본 노트북과 동일),
값의 의미는 대체로  0 = 의심,  -1 = 정상으로 간주,  1 = 피싱 의심 이다.
"""

from __future__ import annotations

import re
import socket
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlsplit

# 네트워크 계열 피처의 기본 타임아웃(초)
TIMEOUT = 3

# 대표적인 URL 단축 서비스 패턴 (원본 노트북 기준)
SHORTENING_SERVICES = (
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"
    r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"
    r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
    r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|db\.tt|"
    r"qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|q\.gs|is\.gd|"
    r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|x\.co|"
    r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|"
    r"tr\.im|link\.zip\.net|buff\.ly|rb\.gy|rebrand\.ly|short\.cm|clk\.im|cutt\.ly|t2m\.io|bl\.ink|"
    r"tiny\.cc"
)

IP_PATTERN = r"(?:\d{1,3}\.){3}\d{1,3}"


# ---------------------------------------------------------------------------
# 1. URL 문자열 기반 피처 (네트워크 불필요)
# ---------------------------------------------------------------------------

def ip_like(url: str) -> int:
    """URL 안에 IP 주소 형식이 들어있으면 1, 아니면 0."""
    return 1 if re.search(IP_PATTERN, url) else 0


def have_at(url: str) -> int:
    """URL 에 '@' 가 있으면 1, 아니면 0."""
    return 1 if "@" in url else 0


def url_depth(url: str) -> int:
    """경로(path) 안의 '/' 로 구분되는 깊이(빈 조각 제외)."""
    segments = urlparse(url).path.split("/")
    return sum(1 for s in segments if len(s) != 0)


def redirection(url: str) -> int:
    """프로토콜(//) 이후에 다시 '//' 가 나오는 리디렉션 의심 여부.

    0 = 의심, -1 = 정상.
    """
    pos = url.rfind("//")
    if pos > 6:
        return 0 if pos > 7 else -1
    return -1


def is_https(url: str) -> int:
    """스킴이 https 면 1, 아니면 0."""
    return 1 if urlsplit(url).scheme == "https" else 0


def tiny_url(url: str) -> int:
    """알려진 단축 URL 서비스 패턴이 포함되어 있으면 1, 아니면 0."""
    return 1 if re.search(SHORTENING_SERVICES, url) else 0


def check_hyphen(url: str) -> int:
    """URL 에 '-' 가 있으면 1, 아니면 0."""
    return 1 if "-" in url else 0


def query_count(url: str) -> int:
    """쿼리 문자열(key=value 쌍)의 개수.

    원본 노트북은 파싱된 dict 를 반환했으나, 학습에 쓰인 값은 '개수'이므로
    여기서는 유효한 key=value 쌍의 개수를 반환한다.
    """
    if "?" not in url:
        return 0
    query_string = url.split("?")[-1]
    count = 0
    for pair in query_string.split("&"):
        if pair and "=" in pair:
            count += 1
    return count


# ---------------------------------------------------------------------------
# 2. WHOIS 기반 피처 (도메인 등록 정보 필요)
# ---------------------------------------------------------------------------

def _netloc(url: str) -> str:
    return urlsplit(url).netloc


def domain_age(url: str) -> int:
    """도메인 생성일 기준. 2년 이상 되었으면 -1(정상), 그 외 0(의심).

    python-whois 패키지가 필요하다.
    """
    import whois  # 지연 임포트: 문자열 피처만 쓸 때 의존성 회피

    try:
        socket.setdefaulttimeout(TIMEOUT)
        info = whois.whois(_netloc(url))
        creation = info.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if creation is None or isinstance(creation, str):
            return 0  # 정보 없음 (의심)
        two_years_ago = datetime.today() - timedelta(days=730)
        return -1 if creation <= two_years_ago else 0
    except Exception:
        return 0


def domain_end(url: str) -> int:
    """도메인 만료일 기준. 만료까지 5개월 이상 남았으면 -1(정상), 그 외 0(의심).

    python-whois 패키지가 필요하다.
    """
    import whois

    try:
        socket.setdefaulttimeout(TIMEOUT)
        info = whois.whois(_netloc(url))
        expiration = info.expiration_date
        if isinstance(expiration, str):
            expiration = datetime.strptime(expiration, "%Y-%m-%d")
        if expiration is None or isinstance(expiration, list):
            return 0
        days_until_expiry = (expiration - datetime.now()).days
        return 0 if (days_until_expiry / 30) < 5 else -1
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# 3. HTTP 요청 기반 피처 (실제 사이트 접속 필요)
# ---------------------------------------------------------------------------

def mouseover(url: str) -> int:
    """페이지에 onmouseover 속성을 가진 태그가 있으면 1, 없으면 0, 요청 실패 시 -1."""
    import requests
    from bs4 import BeautifulSoup

    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return 1 if soup.find(attrs={"onmouseover": True}) else 0
    except requests.exceptions.RequestException:
        return -1


def web_forwards(url: str) -> int:
    """리디렉션(웹 포워딩) 횟수 기반. 3회 이상이면 1(피싱 의심), 2회 이하면 0."""
    import requests

    try:
        resp = requests.get(url, allow_redirects=True, timeout=TIMEOUT)
    except requests.RequestException:
        return 1  # 요청 실패 시 피싱으로 간주
    if not resp.history:
        return 0
    return 0 if len(resp.history) <= 2 else 1


def hyperlinks(url: str) -> int:
    """페이지 내 <a> 태그(하이퍼링크) 개수. 요청 실패 시 -1."""
    import requests
    from bs4 import BeautifulSoup

    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        return len(soup.find_all("a"))
    except (requests.RequestException, requests.Timeout):
        return -1


def domain_consistency(url: str) -> int:
    """요청 후 최종 도메인이 원래 도메인과 같으면 1, 다르면 0, 실패 시 -1."""
    import requests

    try:
        resp = requests.get(url, timeout=5)
        original = urlparse(url).netloc
        final = urlparse(resp.url).netloc
        return 1 if original == final else 0
    except requests.RequestException:
        return -1


# ---------------------------------------------------------------------------
# 4. 텍스트 모델용 토큰화 (CNN / CountVectorizer 입력)
# ---------------------------------------------------------------------------

def tokenize_url(url: str) -> list[str]:
    """URL 을 scheme / 도메인 조각 / 경로 조각으로 분해한 토큰 리스트."""
    parsed = urlparse(url)
    parts = [parsed.scheme] + parsed.netloc.split(".") + parsed.path.split("/")
    return [p for p in parts if p]


def tokenized_url(url: str) -> str:
    """tokenize_url 결과를 공백으로 이어붙인 문자열 (Tokenized_url 컬럼용)."""
    return " ".join(tokenize_url(url))


# ---------------------------------------------------------------------------
# 피처 레지스트리 & 전체 추출
# ---------------------------------------------------------------------------

# 컬럼명 -> 함수. valid_data.csv 의 컬럼 순서와 동일하게 유지한다.
STRING_FEATURES = {
    "IP_LIKE": ip_like,
    "AT": have_at,
    "URL_Depth": url_depth,
    "Redirection": redirection,
    "Is_Https": is_https,
    "TINY_URL": tiny_url,
    "Check_Hyphen": check_hyphen,
    "Query": query_count,
}

WHOIS_FEATURES = {
    "Domain_Age": domain_age,
    "Domain_end": domain_end,
}

NETWORK_FEATURES = {
    "Mouseover": mouseover,
    "Web_forwards": web_forwards,
    "Hyperlinks": hyperlinks,
    "Domain_Cons": domain_consistency,
}


def extract_features(url: str, include_network: bool = True) -> dict:
    """단일 URL 에서 전체 피처를 추출해 dict 로 반환한다.

    include_network=False 로 두면 WHOIS/HTTP 요청이 필요한 피처는 건너뛰고
    URL 문자열 기반 피처와 Tokenized_url 만 계산한다(빠름).
    """
    features: dict = {}
    for name, fn in STRING_FEATURES.items():
        features[name] = fn(url)
    if include_network:
        for name, fn in {**WHOIS_FEATURES, **NETWORK_FEATURES}.items():
            features[name] = fn(url)
    features["Tokenized_url"] = tokenized_url(url)
    return features


if __name__ == "__main__":
    # 문자열 기반 피처만 빠르게 확인
    demo = "https://gabiaing.com/login?redirect=home&id=1"
    print(f"URL: {demo}")
    for k, v in extract_features(demo, include_network=False).items():
        print(f"  {k:14s} = {v}")
