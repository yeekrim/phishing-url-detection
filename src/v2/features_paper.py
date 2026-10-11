"""v2 피처 규칙 — Mohammad, Thabtah, McCluskey "Phishing Websites Features" 문서 기준.

원 출처: Mohammad, McCluskey, Thabtah, "An Assessment of Features Related to Phishing Websites
using an Automated Technique", ICITST 2012 / UCI Phishing Websites dataset (id 327).

값 체계는 UCI 데이터셋과 같다:  1 = 정상(Legitimate),  0 = 의심(Suspicious),  -1 = 피싱(Phishing).
네트워크 정보가 없어 판단할 수 없으면 None(빈 값)을 돌려준다. 2024년처럼 실패를 -1 로 채우면
"접속 실패 = 피싱"이라는 지름길이 생기기 때문이다.

논문에 없는 피처(URL_Depth, Query, Hyperlinks, Domain_Cons)는 2024년 기획의 정의를 따르되,
규칙으로 바꾸지 않고 원래 숫자를 그대로 저장한다.

이 모듈의 함수는 모두 네트워크를 쓰지 않는다. 네트워크 조회 결과(리디렉션 수, HTML, 인증서,
WHOIS 날짜)는 fetch.py 가 구해서 넘겨준다. 그래서 단위 테스트로 규칙을 그대로 검증할 수 있다.
"""

from __future__ import annotations

import ipaddress
import re
from datetime import datetime
from urllib.parse import urlsplit

LEGIT, SUSPICIOUS, PHISHING = 1, 0, -1

# 2024년 코드와 같은 단축 URL 목록
SHORTENING_SERVICES = re.compile(
    r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"
    r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"
    r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
    r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|db\.tt|"
    r"qr\.ae|adf\.ly|bitly\.com|cur\.lv|tinyurl\.com|ity\.im|q\.gs|"
    r"po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|"
    r"prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|"
    r"link\.zip\.net|buff\.ly|rb\.gy|rebrand\.ly|short\.cm|clk\.im|cutt\.ly|t2m\.io|bl\.ink",
    re.IGNORECASE,
)

# 논문이 꼽은 신뢰 인증서 발급사 (2012년 기준 목록 그대로)
TRUSTED_ISSUERS = ("GeoTrust", "GoDaddy", "Network Solutions", "Thawte", "Comodo", "Doster", "VeriSign")

HEX_IP = re.compile(r"^(0x[0-9a-f]{1,2}\.){3}0x[0-9a-f]{1,2}$", re.IGNORECASE)
STATUS_BAR = re.compile(r"window\.status|status\s*=", re.IGNORECASE)


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


# ---------------------------------------------------------------------------
# 주소창 기반 (논문: Address Bar based Features)
# ---------------------------------------------------------------------------

def ip_like(url: str) -> int:
    """도메인 자리에 IP 주소(16진수 표기 포함)를 쓰면 피싱."""
    host = _host(url)
    try:
        ipaddress.ip_address(host)
        return PHISHING
    except ValueError:
        return PHISHING if HEX_IP.match(host) else LEGIT


def have_at(url: str) -> int:
    """URL 에 '@' 가 있으면 피싱."""
    return PHISHING if "@" in url else LEGIT


def redirection(url: str) -> int:
    """마지막 '//' 의 위치가 7보다 뒤면 피싱.

    논문은 http 면 6번째, https 면 7번째 자리에 '//' 가 온다고 설명하므로 위치를 1부터 센다.
    """
    position = url.rfind("//") + 1
    return PHISHING if position > 7 else LEGIT


def tiny_url(url: str) -> int:
    """단축 URL 서비스를 쓰면 피싱."""
    return PHISHING if SHORTENING_SERVICES.search(_host(url)) else LEGIT


def prefix_suffix(url: str) -> int:
    """도메인 부분에 '-' 가 있으면 피싱. (기획 컬럼명: Check_Hyphen)"""
    return PHISHING if "-" in _host(url) else LEGIT


def https_rule(url: str, cert: dict | None) -> int | None:
    """https 를 쓰고, 발급사가 신뢰 목록에 있고, 인증서가 1년 이상 되었으면 정상.
    https 지만 발급사가 신뢰 목록에 없으면 의심. 그 외(https 아님 등)는 피싱.

    cert: fetch.get_certificate() 결과 {"issuer": str, "not_before": datetime, "verified": bool}
          https URL 인데 인증서를 못 가져왔으면 None → 판단 불가(None).
    """
    if urlsplit(url).scheme.lower() != "https":
        return PHISHING
    if cert is None:
        return None
    trusted = cert["verified"] and any(name.lower() in cert["issuer"].lower() for name in TRUSTED_ISSUERS)
    if not trusted:
        return SUSPICIOUS
    age_days = (cert["checked_at"] - cert["not_before"]).days
    return LEGIT if age_days >= 365 else PHISHING


# ---------------------------------------------------------------------------
# 도메인 기반 (논문: Domain based Features)
# ---------------------------------------------------------------------------

def domain_age(created: datetime | None, checked_at: datetime) -> int | None:
    """도메인 생성 후 6개월 이상이면 정상, 아니면 피싱."""
    if created is None:
        return None
    return LEGIT if (checked_at - created).days >= 182 else PHISHING


def registration_length(expires: datetime | None, checked_at: datetime) -> int | None:
    """만료까지 1년 이하로 남았으면 피싱, 아니면 정상. (기획 컬럼명: Domain_end)"""
    if expires is None:
        return None
    return PHISHING if (expires - checked_at).days <= 365 else LEGIT


# ---------------------------------------------------------------------------
# HTML·JavaScript 기반 (논문: HTML and JavaScript based Features)
# ---------------------------------------------------------------------------

def website_forwarding(n_redirects: int | None) -> int | None:
    """리디렉션 1회 이하 정상, 2~3회 의심, 4회 이상 피싱."""
    if n_redirects is None:
        return None
    if n_redirects <= 1:
        return LEGIT
    return SUSPICIOUS if n_redirects < 4 else PHISHING


def status_bar_customization(onmouseover_values: list[str] | None) -> int | None:
    """onMouseOver 가 상태표시줄을 바꾸면 피싱. (기획 컬럼명: Mouseover)"""
    if onmouseover_values is None:
        return None
    return PHISHING if any(STATUS_BAR.search(v) for v in onmouseover_values) else LEGIT


# ---------------------------------------------------------------------------
# 논문에 없는 피처 (2024년 기획에서 추가) — 원래 숫자를 그대로 저장
# ---------------------------------------------------------------------------

def url_depth(url: str) -> int:
    """경로의 '/' 깊이 (빈 조각 제외)."""
    return sum(1 for s in urlsplit(url).path.split("/") if s)


def query_count(url: str) -> int:
    """쿼리 문자열의 key=value 쌍 개수."""
    query = urlsplit(url).query
    return sum(1 for pair in query.split("&") if pair and "=" in pair)


def hyperlink_count(n_anchors: int | None) -> int | None:
    """페이지의 <a> 태그 수."""
    return n_anchors


def domain_consistency(url: str, final_url: str | None) -> int | None:
    """리디렉션을 따라간 최종 호스트가 처음 호스트와 같으면 1, 다르면 0."""
    if final_url is None:
        return None
    return 1 if _host(url) == _host(final_url) else 0


# ---------------------------------------------------------------------------
# 텍스트 모델용 토큰 (2024년과 같음)
# ---------------------------------------------------------------------------

def tokenized_url(url: str) -> str:
    parts = urlsplit(url)
    tokens = [parts.scheme] + (parts.hostname or "").split(".") + parts.path.split("/")
    return " ".join(t for t in tokens if t)


STRING_FEATURES = ["IP_LIKE", "AT", "URL_Depth", "Redirection", "TINY_URL", "Check_Hyphen", "Query"]
WHOIS_FEATURES = ["Domain_Age", "Domain_end"]
NETWORK_FEATURES = ["Is_Https", "Mouseover", "Web_forwards", "Hyperlinks", "Domain_Cons"]
FEATURE_COLUMNS = STRING_FEATURES + WHOIS_FEATURES + NETWORK_FEATURES


def compute(url: str, page: dict | None, cert: dict | None, whois: dict | None) -> dict:
    """한 URL 의 피처 14개 + 토큰열.

    page:  fetch.fetch_page() 결과 {"final_url", "n_redirects", "n_anchors", "onmouseover"} 또는 None
    cert:  fetch.get_certificate() 결과 또는 None
    whois: fetch.lookup_whois() 결과 {"created", "expires", "checked_at"} 또는 None
    """
    page = page or {}
    whois = whois or {}
    checked_at = whois.get("checked_at")
    return {
        "IP_LIKE": ip_like(url),
        "AT": have_at(url),
        "URL_Depth": url_depth(url),
        "Redirection": redirection(url),
        "TINY_URL": tiny_url(url),
        "Check_Hyphen": prefix_suffix(url),
        "Query": query_count(url),
        "Domain_Age": domain_age(whois.get("created"), checked_at) if checked_at else None,
        "Domain_end": registration_length(whois.get("expires"), checked_at) if checked_at else None,
        "Is_Https": https_rule(url, cert),
        "Mouseover": status_bar_customization(page.get("onmouseover")),
        "Web_forwards": website_forwarding(page.get("n_redirects")),
        "Hyperlinks": hyperlink_count(page.get("n_anchors")),
        "Domain_Cons": domain_consistency(url, page.get("final_url")),
        "Tokenized_url": tokenized_url(url),
    }
