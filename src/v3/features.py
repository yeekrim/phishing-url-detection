"""v3 피처 — URL 문자열 + HTML 구조 + 브랜드-도메인 일치(BDI). 네트워크를 쓰지 않는 순수 함수.

근거
    - URL 피처: 기존 연구에서 널리 쓰는 URL 어휘 피처(길이, 특수문자, 하위 도메인, IP, 퓨니코드 등).
      PhreshPhish(Dalton 외, 2025)의 기준 모델도 "URL 길이 같은 손으로 만든 피처 + 글자 n-gram" 을 쓴다.
    - HTML 피처: Çolhak 외(arXiv 2401.04820)의 C3~C13. 논문에 이름만 있고 정의가 없는 것
      (C6~C9, C11~C13)은 이름에 맞게 직접 정의했다 (각 함수 주석에 표시).
    - BDI 피처: Mishra & Varshney(arXiv 2503.06487). 페이지 도메인과 같으면 1, 다르면 -1, 없으면 0.
      논문의 5개 중 HTML 만으로 계산할 수 있는 3개(로고 도메인, 폼 전송 도메인, 가장 흔한 링크 도메인)만.
      인증서 CN·쿠키 도메인은 실제 접속이 필요해서 뺀다.

도메인 비교는 공개 접미사 목록의 사설 항목(github.io 등)까지 반영한 등록 도메인으로 한다.
a.github.io 와 b.github.io 는 운영자가 다르기 때문이다.

URL 피처는 쿼리(?...)와 프래그먼트(#...)를 지운 주소로 계산한다. PhreshPhish 는 개인정보 보호를 위해
쿼리를 지웠는데, 그 작업을 정상 클래스에만 했다(논문 부록 A.2: "We focused our PII removal efforts on the
benign class only"). 실제로 시험 파일에서 쿼리가 있는 URL 은 정상 0% / 피싱 6.8% 였다. 그대로 두면
"쿼리가 있으면 피싱" 이라는 지름길이 생기므로 두 클래스 모두 똑같이 지운다.
같은 이유로 URL 은 소문자로 바꿔 계산한다. 대문자가 있는 URL 비율이 정상은 train/test 모두 13% 인데
피싱은 train 17% / test 2% 로, 수집 묶음에 따라 피싱 URL 의 대소문자가 바뀌어 저장된 것으로 보인다.

HTML 은 신뢰할 수 없는 데이터다. lxml 로 텍스트 파싱만 하고, 스크립트 실행·외부 자원 요청은 하지 않는다.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from urllib.parse import urljoin, urlsplit

import lxml.html
import tldextract
from lxml import etree

_extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)
_extract_public = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

MAX_HTML_CHARS = 2_000_000
TITLE_CHARS, TEXT_CHARS, HTML_PREFIX_CHARS = 200, 2000, 5000

SHORTENERS = {"bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly", "rb.gy", "cutt.ly",
              "rebrand.ly", "shorturl.at", "tiny.cc", "s.id", "t.ly", "bit.do", "qrco.de", "q-r.to"}
SUSPICIOUS_WORDS = ("login", "log-in", "signin", "sign-in", "verify", "verification", "secure", "account",
                    "update", "confirm", "banking", "password", "wallet", "support", "unlock", "suspend",
                    "billing", "invoice", "auth", "webscr", "recover")
OBFUSCATION = re.compile(r"eval\s*\(|unescape\s*\(|atob\s*\(|fromCharCode|document\.write\s*\(|"
                         r"[A-Za-z0-9+/=]{300,}|(?:\\x[0-9a-fA-F]{2}){20,}")
COPYRIGHT = re.compile(r"©|&copy;|copyright", re.IGNORECASE)
NULL_HREF = re.compile(r"^\s*(#.*|javascript:.*|about:blank)?\s*$", re.IGNORECASE)


def registered_domain(host: str) -> str:
    ext = _extract(host)
    return (ext.top_domain_under_public_suffix or host or "").lower()


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    return -sum(c / len(s) * math.log2(c / len(s)) for c in counts.values())


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# URL
# ---------------------------------------------------------------------------

def safe_split(url: str):
    """urlsplit 이 형식 오류(예: 대괄호 안이 IP 가 아닌 호스트 '[abc]')로 실패하면 대괄호를 빼고 다시 나눈다."""
    try:
        return urlsplit(url)
    except ValueError:
        return urlsplit(url.replace("[", "").replace("]", ""))


def _port(parts) -> int | None:
    try:
        return parts.port
    except ValueError:  # 숫자가 아니거나 범위를 벗어난 포트
        return -1


def strip_query(url: str) -> str:
    """쿼리와 프래그먼트를 지우고 소문자로 바꾼 주소 (두 클래스에 똑같이 적용)."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return url.split("?")[0].split("#")[0].lower()
    return parts._replace(query="", fragment="").geturl().lower()


def url_features(url: str) -> dict:
    """url 은 strip_query() 를 거친 주소여야 한다."""
    parts = safe_split(url)
    host = (parts.hostname or "").lower()
    ext = _extract(host)
    public = _extract_public(host)
    path = parts.path
    lower = url.lower()
    host_tokens = re.split(r"[.\-]", host)
    return {
        "url_len": len(url),
        "host_len": len(host),
        "path_len": len(path),
        "n_dots_host": host.count("."),
        "n_hyphens_host": host.count("-"),
        "n_digits_host": sum(ch.isdigit() for ch in host),
        "n_digits_url": sum(ch.isdigit() for ch in url),
        "digit_ratio_url": sum(ch.isdigit() for ch in url) / max(len(url), 1),
        "n_special_url": sum(url.count(c) for c in "@~%=&?_!*,;$+"),
        "n_subdomain_levels": len([s for s in ext.subdomain.split(".") if s]),
        "path_depth": len([s for s in path.split("/") if s]),
        "has_ip": int(_is_ip(host)),
        "has_port": int(_port(parts) is not None),
        "is_https": int(parts.scheme == "https"),
        "has_at": int("@" in url),
        "has_double_slash_path": int("//" in path),
        "has_punycode": int("xn--" in host),
        "host_entropy": _entropy(host),
        "url_entropy": _entropy(url),
        "longest_host_token": max((len(t) for t in host_tokens), default=0),
        "has_www": int(host.startswith("www.")),
        "is_shortener": int(public.top_domain_under_public_suffix in SHORTENERS),
        # 사설 접미사(github.io, weebly.com 등) 아래의 사이트 = 무료 호스팅·플랫폼
        "is_hosted_platform": int(bool(ext.is_private)),
        "n_suspicious_words": sum(w in lower for w in SUSPICIOUS_WORDS),
        "has_php": int(".php" in path.lower()),
        "has_html_ext": int(path.lower().endswith((".html", ".htm"))),
        "tld": (public.suffix or "").lower(),
    }


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def _domain_of(href: str, base_url: str) -> str | None:
    """링크가 가리키는 등록 도메인. http(s) 가 아니면 None."""
    try:
        absolute = urljoin(base_url, href.strip())
        parts = urlsplit(absolute)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    return registered_domain(parts.hostname)


def _match(domain: str | None, page_domain: str) -> int:
    """BDI 값: 같으면 1, 다르면 -1, 없으면 0."""
    if domain is None:
        return 0
    return 1 if domain == page_domain else -1


EMPTY_HTML = {
    "html_len": 0, "n_tags": 0, "n_links": 0, "n_internal_links": 0, "n_external_links": 0,
    "n_null_links": 0, "external_link_ratio": 0.0, "null_link_ratio": 0.0, "self_link_ratio": 0.0,
    "n_css": 0, "n_external_css": 0, "n_js_src": 0, "n_external_js": 0, "n_inline_js": 0,
    "inline_js_len": 0, "n_obfuscated_js": 0, "n_forms": 0, "n_inputs": 0, "n_password_inputs": 0,
    "n_hidden_inputs": 0, "n_email_inputs": 0, "suspicious_form_link": 0, "n_iframes": 0,
    "n_images": 0, "has_footer": 0, "has_copyright": 0, "has_meta_description": 0,
    "has_meta_refresh": 0, "has_favicon": 0, "external_favicon": 0, "title_len": 0, "text_len": 0,
    "n_words": 0, "bdi_logo": 0, "bdi_form_action": 0, "bdi_common_link": 0,
}


def html_features(html: str | None, url: str) -> tuple[dict, dict]:
    """(숫자 피처, 텍스트 필드). 파싱할 수 없으면 숫자 피처는 0, 텍스트는 빈 문자열."""
    texts = {"title": "", "text": "", "html_prefix": (html or "")[:HTML_PREFIX_CHARS]}
    feats = dict(EMPTY_HTML)
    if not html or not html.strip():
        return feats, texts
    html = html[:MAX_HTML_CHARS]
    feats["html_len"] = len(html)
    try:
        doc = lxml.html.document_fromstring(html)
    except (etree.ParserError, ValueError):
        return feats, texts

    page_domain = registered_domain((safe_split(url).hostname or "").lower())
    feats["n_tags"] = sum(1 for _ in doc.iter(etree.Element))

    # C3~C5 링크 수 + 빈 링크
    link_domains = []
    n_internal = n_external = n_null = n_self = 0
    for a in doc.iter("a"):
        href = a.get("href")
        if href is None:
            continue
        if NULL_HREF.match(href):
            n_null += 1
            n_self += 1
            continue
        d = _domain_of(href, url)
        if d is None:
            continue
        link_domains.append(d)
        if d == page_domain:
            n_internal += 1
            if urljoin(url, href).split("#")[0] == url.split("#")[0]:
                n_self += 1
        else:
            n_external += 1
    n_links = n_internal + n_external + n_null
    feats.update(n_links=n_links, n_internal_links=n_internal, n_external_links=n_external, n_null_links=n_null,
                 external_link_ratio=n_external / max(n_links, 1), null_link_ratio=n_null / max(n_links, 1),
                 # C11 common_page_ratio (논문에 정의 없음): 같은 페이지(#, 빈 링크, 자기 주소)를 가리키는 링크 비율
                 self_link_ratio=n_self / max(n_links, 1))

    # C6~C8 외부 CSS·JS (정의 없음): 다른 등록 도메인에서 불러오는 파일 수
    for link in doc.iter("link"):
        rel = (link.get("rel") or "").lower()
        href = link.get("href") or ""
        if "stylesheet" in rel:
            feats["n_css"] += 1
            d = _domain_of(href, url)
            feats["n_external_css"] += int(d is not None and d != page_domain)
        if "icon" in rel and href:
            feats["has_favicon"] = 1
            d = _domain_of(href, url)
            feats["external_favicon"] = int(d is not None and d != page_domain)
    for script in doc.iter("script"):
        src = script.get("src")
        if src:
            feats["n_js_src"] += 1
            d = _domain_of(src, url)
            feats["n_external_js"] += int(d is not None and d != page_domain)
        else:
            code = script.text or ""
            feats["n_inline_js"] += 1
            feats["inline_js_len"] += len(code)
            # C9 obfuscated_js_count (정의 없음): eval/atob/fromCharCode, 긴 base64·\x 문자열이 있는 인라인 스크립트
            feats["n_obfuscated_js"] += int(bool(OBFUSCATION.search(code)))

    # 폼 + C10 suspicious_form_link: 비밀번호 입력란이 있는 폼이 외부·빈 주소·메일로 전송
    form_domain = None
    for form in doc.iter("form"):
        feats["n_forms"] += 1
        action = (form.get("action") or "").strip()
        has_password = any((i.get("type") or "").lower() == "password" for i in form.iter("input"))
        if action.lower().startswith(("mailto:", "javascript:", "about:")):
            target = "external"
        else:
            d = _domain_of(action, url) if action else page_domain
            target = "self" if d == page_domain else "external"
            if form_domain is None:
                form_domain = d
        if has_password and (target == "external" or action in ("", "#")):
            feats["suspicious_form_link"] = 1
    for inp in doc.iter("input"):
        kind = (inp.get("type") or "text").lower()
        feats["n_inputs"] += 1
        feats["n_password_inputs"] += int(kind == "password")
        feats["n_hidden_inputs"] += int(kind == "hidden")
        feats["n_email_inputs"] += int(kind == "email" or "mail" in (inp.get("name") or "").lower())

    feats["n_iframes"] = sum(1 for _ in doc.iter("iframe"))

    # 로고: src·alt·class·id 에 'logo' 가 들어간 이미지 중 첫 번째
    logo_domain = None
    for img in doc.iter("img"):
        feats["n_images"] += 1
        attrs = " ".join(img.get(k) or "" for k in ("src", "alt", "class", "id")).lower()
        if logo_domain is None and "logo" in attrs and img.get("src"):
            logo_domain = _domain_of(img.get("src"), url)

    for meta in doc.iter("meta"):
        name = (meta.get("name") or "").lower()
        equiv = (meta.get("http-equiv") or "").lower()
        if name == "description" and (meta.get("content") or "").strip():
            feats["has_meta_description"] = 1  # C13
        if equiv == "refresh":
            feats["has_meta_refresh"] = 1

    title_el = doc.find(".//title")
    title = (title_el.text_content() if title_el is not None else "").strip()
    for bad in doc.xpath("//script|//style|//noscript"):
        bad.drop_tree()
    body_text = " ".join(doc.text_content().split())
    feats.update(
        # C12 common_page_footer (정의 없음): <footer> 태그나 저작권 표기
        has_footer=int(doc.find(".//footer") is not None),
        has_copyright=int(bool(COPYRIGHT.search(body_text))),
        title_len=len(title), text_len=len(body_text), n_words=len(body_text.split()),
        bdi_logo=_match(logo_domain, page_domain),
        bdi_form_action=_match(form_domain, page_domain),
        bdi_common_link=_match(Counter(link_domains).most_common(1)[0][0] if link_domains else None, page_domain),
    )
    texts.update(title=title[:TITLE_CHARS], text=body_text[:TEXT_CHARS])
    return feats, texts


def compute(url: str, html: str | None) -> dict:
    """한 행의 전체 피처 + 텍스트 필드. url_clean 은 URL 문자열 모델(n-gram)에도 이 주소를 쓰기 위해 저장."""
    clean = strip_query(url)
    feats, texts = html_features(html, url)
    return {"url_clean": clean, **url_features(clean), **feats, **texts}
