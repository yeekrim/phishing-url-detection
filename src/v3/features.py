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
같은 이유로 URL 은 소문자로 바꾸고 끝의 '/' 를 지운다. 경로가 없는 URL(https://a.com)이 정상 0% /
피싱 train 6% · test 42% 였다 — 정상 URL 은 항상 'https://a.com/' 꼴로 저장되어 있었다.
HTML 맨 앞의 <!DOCTYPE> 선언도 지운다. 정상은 99% 이상이 선언 없이 '<html' 로 시작하는데 피싱은
train 0% / test 75% 가 선언으로 시작해, 수집 시기에 따라 저장 방식이 달랐던 것으로 보인다.
URL 대소문자도 대문자가 있는 URL 비율이 정상은 train/test 모두 13% 인데
피싱은 train 17% / test 2% 로, 수집 묶음에 따라 피싱 URL 의 대소문자가 바뀌어 저장된 것으로 보인다.

HTML 은 신뢰할 수 없는 데이터다. lxml 로 텍스트 파싱만 하고, 스크립트 실행·외부 자원 요청은 하지 않는다.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import NamedTuple
from urllib.parse import urljoin, urlsplit

import lxml.html
import tldextract
from lxml import etree

_extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None, include_psl_private_domains=True)
_extract_public = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

MAX_HTML_CHARS = 2_000_000
TITLE_CHARS, TEXT_CHARS, HTML_PREFIX_CHARS = 200, 2000, 5000
FULL_TEXT_CHARS = 200_000

SHORTENERS = {"bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly", "rb.gy", "cutt.ly",
              "rebrand.ly", "shorturl.at", "tiny.cc", "s.id", "t.ly", "bit.do", "qrco.de", "q-r.to"}
SUSPICIOUS_WORDS = ("login", "log-in", "signin", "sign-in", "verify", "verification", "secure", "account",
                    "update", "confirm", "banking", "password", "wallet", "support", "unlock", "suspend",
                    "billing", "invoice", "auth", "webscr", "recover")
OBFUSCATION = re.compile(r"eval\s*\(|unescape\s*\(|atob\s*\(|fromCharCode|document\.write\s*\(|"
                         r"[A-Za-z0-9+/=]{300,}|(?:\\x[0-9a-fA-F]{2}){20,}")
COPYRIGHT = re.compile(r"©|&copy;|copyright", re.IGNORECASE)
NULL_HREF = re.compile(r"^\s*(#.*|javascript:.*|about:blank)?\s*$", re.IGNORECASE)


# 공개 접미사 목록(tldextract 5.4.0 내장)에 사설 항목으로 없는 호스팅·사이트 빌더 서비스.
# 하위 도메인마다 운영자가 다른 서비스만 넣었다 (서비스 성격으로 골랐고, 피싱 비율로 고르지 않았다).
# 근거: PhreshPhish 전체에서 하위 도메인이 20개 이상인 등록 도메인 412개를 하나씩 보고 고름.
# 무작위 하위 도메인을 쓰는 공격자 도메인(kleinanzegien.de 등), 메일 추적·단축 링크(sendgrid.net, page.link 등)는 넣지 않았다.
# 예: test 의 *.blogspot.<국가> 피싱 17,848건 중 17,501건이 플랫폼으로 인식되지 않았고, gitbook.io 하위 도메인 19,547개도 마찬가지였다.
EXTRA_PLATFORMS = {
    "gitbook.io", "weeblysite.com", "weebly.com", "godaddysites.com", "fandom.com", "surge.sh", "webcindario.com",
    "studio.site", "mybluehost.me", "teachable.com", "webwave.dev", "ubpages.com", "activehosted.com",
    "codeanyapp.com", "mystrikingly.com", "myqcloud.com", "serv00.net", "wcomhost.com", "hubside.fr", "windows.net",
    "formstack.com", "sviluppo.host", "zendesk.com", "substack.com", "atlassian.net", "mmm.page", "wordpress.com",
    "hs-sites.com", "backblazeb2.com", "edgeone.app", "flazio.com", "squarespace.com", "teemill.com",
    "crazydomains.com", "tumblr.com", "zohodesk.com", "daftpage.com", "hostingersite.com", "medium.com",
    "mytemp.website", "freewebhostmost.com", "myportfolio.com", "000.pe", "forms.app", "filesusr.com", "rf.gd",
    "b-cdn.net", "mypixieset.com", "000webhostapp.com", "freshdesk.com", "odoo.com", "4everland.app", "free.nf",
    "ct.ws", "temporary.site", "gorgias.help", "systeme.io", "fleek.co", "b12sites.com", "salesforce-sites.com",
    "tempisite.com", "tiiny.site", "inmotionhosting.com", "pory.app", "rollout.site", "zya.me", "itch.io",
    "peraichi.com", "jimdofree.com", "myftpupload.com", "gumroad.com", "statuspage.io", "webnode.page",
    "temporary-demo.site", "webydo.com", "myclickfunnels.com", "secureserver.net", "sharepoint.com",
}


class HostParts(NamedTuple):
    subdomain: str
    domain: str      # 사이트 이름 (등록 도메인의 접미사 앞 부분)
    suffix: str
    is_private: bool  # 호스팅 플랫폼 아래의 사이트인가


def split_host(host: str) -> HostParts:
    """tldextract(사설 접미사 포함) + 내장 목록에 없는 플랫폼(EXTRA_PLATFORMS, 국가별 blogspot) 보정."""
    host = (host or "").lower()
    ext = _extract(host)
    if not ext.is_private:
        base = _extract_public(host).top_domain_under_public_suffix
        is_platform = base in EXTRA_PLATFORMS or base.startswith("blogspot.")
        rest = host[: -(len(base) + 1)] if base and host.endswith("." + base) else ""
        if is_platform and rest and rest != "www":
            parts = rest.split(".")
            return HostParts(".".join(parts[:-1]), parts[-1], base, True)
    return HostParts(ext.subdomain, ext.domain, ext.suffix, bool(ext.is_private))


def registered_domain(host: str) -> str:
    parts = split_host(host)
    if parts.is_private and parts.domain:
        return f"{parts.domain}.{parts.suffix}"
    return (_extract(host).top_domain_under_public_suffix or host or "").lower()


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
    """쿼리·프래그먼트·끝의 '/' 를 지우고 소문자로 바꾼 주소 (두 클래스에 똑같이 적용)."""
    try:
        parts = urlsplit(url)
        clean = parts._replace(query="", fragment="").geturl()
    except ValueError:
        clean = url.split("?")[0].split("#")[0]
    return clean.lower().rstrip("/")


def mask_jp(url: str) -> str:
    """URL n-gram 용: 공개 접미사가 .jp 로 끝나면(.jp, .co.jp, .go.jp …) .com 으로 바꾼다.
    train 에서 .jp 는 정상 8,925 / 피싱 91 개라 모델이 ".jp 면 정상" 으로 배웠고, test .jp 피싱 탐지율이
    61.5% 로 그 외(76.2%)보다 낮았다. 실제 공격자가 해킹한 .jp 사이트에 피싱을 올리면 그대로 통과한다.
    .com 으로 바꾸는 이유: 정상·피싱 모두에 가장 흔한 접미사라 새 단서가 되지 않는다
    (".tld" 같은 표시로 바꾸면 그 표시가 다시 .jp 역할을 한다)."""
    parts = safe_split(url)
    host = parts.hostname or ""
    suffix = _extract_public(host).suffix
    if not (suffix == "jp" or suffix.endswith(".jp")):
        return url
    return url.replace(host, host[: -len(suffix)] + "com", 1)


DOCTYPE = re.compile(r"^\s*<!doctype[^>]*>\s*", re.IGNORECASE)


def strip_doctype(html: str) -> str:
    return DOCTYPE.sub("", html, count=1)


def url_features(url: str) -> dict:
    """url 은 strip_query() 를 거친 주소여야 한다."""
    parts = safe_split(url)
    host = (parts.hostname or "").lower()
    ext = split_host(host)
    public = _extract_public(host)
    path = parts.path
    lower = url.lower()
    host_tokens = re.split(r"[.\-]", host)
    # 사이트 이름: 등록 도메인의 접미사 앞 부분 (paypal.com → paypal, abc.netlify.app → abc).
    # 호스팅 플랫폼이라는 사실만으로는 정상·피싱을 가를 수 없어서, 이름 자체의 모양을 따로 본다.
    label = "" if _is_ip(host) else ext.domain
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
        "site_label_len": len(label),
        "site_label_digits": sum(ch.isdigit() for ch in label),
        "site_label_hyphens": label.count("-"),
        "site_label_entropy": _entropy(label),
        "site_label_suspicious": int(any(w in label for w in SUSPICIOUS_WORDS)),
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


def html_features(html: str | None, url: str, text_chars: int = TEXT_CHARS) -> tuple[dict, dict]:
    """(숫자 피처, 텍스트 필드). 파싱할 수 없으면 숫자 피처는 0, 텍스트는 빈 문자열.
    text_chars: 본문을 몇 자까지 남길지 (n-gram 용 본문 전체는 FULL_TEXT_CHARS)."""
    texts = {"title": "", "text": "", "html_prefix": strip_doctype(html or "")[:HTML_PREFIX_CHARS]}
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
    texts.update(title=title[:TITLE_CHARS], text=body_text[:text_chars])
    return feats, texts


# ---------------------------------------------------------------------------
# 브랜드-도메인 일치 (제목·본문)
# ---------------------------------------------------------------------------
# Mishra & Varshney(BDI)는 로고·폼·링크의 도메인을 페이지 도메인과 비교한다. 여기서는 같은 생각을
# 페이지 제목·본문으로 넓혀, "페이지가 스스로 밝히는 이름이 이 도메인의 이름인가" 를 본다.
# 브랜드 목록 없이 계산하므로 학습 때 못 본 공격 대상에도 쓸 수 있다.
# 근거(최종 모델 오탐률 0.1%): 비밀번호 입력란이 있는 피싱 탐지율 67%, 점수 최상위 오탐에 공식 사이트
# (correios.com.br, banco.bradesco 등)가 많았다. train 일부에서 "제목 단어가 사이트 이름에 있음" 은
# 정상 61% / 피싱 23%, "철자만 비슷함(0.8 이상)" 은 정상 2% / 피싱 8% 였다.

# 사이트 이름이 아니어도 제목·도메인에 흔히 나오는 단어 (이것끼리 맞는 것은 일치로 보지 않는다)
GENERIC_WORDS = {
    "login", "logon", "signin", "sign", "account", "accounts", "home", "homepage", "welcome", "page", "online",
    "secure", "security", "verify", "verification", "update", "mail", "email", "webmail", "portal", "official",
    "site", "website", "service", "services", "support", "help", "index", "default", "untitled", "loading",
    "document", "file", "files", "error", "bank", "banking", "shop", "store", "free", "news", "blog", "with",
    "your", "from", "this", "that", "here", "best", "about", "contact", "click", "form", "forms", "access",
    "auth", "user", "client", "customer", "member", "members", "center", "centre", "info", "online", "app",
    "apps", "cloud", "server", "host", "hosting", "domain", "http", "https", "www", "html", "index",
}
ACRONYM_SKIP = {"of", "the", "and", "de", "du", "la", "le", "des", "da", "do", "di", "del", "et", "y", "e"}
LOOKALIKE_SIM = 0.8


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _site_label(url: str) -> str:
    host = (safe_split(url).hostname or "").lower()
    return "" if _is_ip(host) else split_host(host).domain.lower()


def brand_features(title: str, text: str, url: str, brands: frozenset[str] = frozenset()) -> dict:
    """title·text 는 저장된 텍스트 필드(제목, 본문 앞 2,000자), url 은 strip_query() 를 거친 주소.
    brands: train 피싱의 공격 대상 이름(정규화). 비어 있으면 known_brand_mismatch 는 0."""
    label = _site_label(url)
    label_flat = re.sub(r"[^a-z0-9]", "", label)
    label_parts = [t for t in re.split(r"[^a-z0-9]+|\d+", label) if len(t) >= 3 and t not in GENERIC_WORDS]
    words = _words(title)
    title_flat = "".join(words)
    tokens = [w for w in words if len(w) >= 4 and w not in GENERIC_WORDS]
    acronym = "".join(w[0] for w in words if w not in ACRONYM_SKIP)
    match = bool(label_flat) and (
        any(t in label_flat for t in tokens)
        or (len(label_flat) >= 4 and label_flat not in GENERIC_WORDS and label_flat in title_flat)
        or (len(label_flat) >= 2 and acronym == label_flat)
    )
    sim = max((SequenceMatcher(None, t, l).ratio() for t in tokens[:15] for l in label_parts + [label_flat]
               if l), default=0.0)
    mentions = (text or "").lower().count(label_flat) if len(label_flat) >= 4 else 0
    named = [b for b in brands if b in title_flat]
    return {
        "title_domain_match": int(match),
        "title_domain_sim": sim,
        "title_lookalike": int(not match and sim >= LOOKALIKE_SIM),
        "text_domain_mentions": min(mentions, 50),
        "known_brand_mismatch": int(bool(named) and not any(b in label_flat for b in named)),
    }


def compute(url: str, html: str | None) -> dict:
    """한 행의 전체 피처 + 텍스트 필드. url_clean 은 URL 문자열 모델(n-gram)에도 이 주소를 쓰기 위해 저장.
    known_brand_mismatch 는 train 의 브랜드 목록이 필요해서 여기서는 0 이고, extract_phreshphish.py --refresh-brand 로 채운다."""
    clean = strip_query(url)
    feats, texts = html_features(html, url)
    return {"url_clean": clean, **url_features(clean), **feats, **texts,
            **brand_features(texts["title"], texts["text"], clean)}
