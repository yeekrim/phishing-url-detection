"""v2 네트워크 조회 — 피싱 사이트에 접속하므로 안전장치를 둔다.

Docker 컨테이너 안에서 실행하는 것을 전제로 한다 (docker/run.sh).

안전장치
    - HTML 텍스트만 받는다. JavaScript 를 실행하지 않고, 폼을 제출하지 않는다.
    - Content-Type 이 text/html 이 아니면 본문을 읽지 않는다.
    - 본문은 최대 2MB 까지만 읽고, 받은 HTML 은 저장하지 않는다 (필요한 숫자만 뽑고 버림).
    - 연결 5초 / 읽기 10초 제한, 리디렉션은 최대 10번까지 직접 따라간다.
    - 리디렉션의 매 단계마다 목적지 IP 를 확인해 사설·루프백·링크로컬 주소면 멈춘다.
      (피싱 URL 이 집 공유기나 내부망 주소로 보내는 경우를 막기 위함)
    - http/https 이외의 주소로는 이동하지 않는다.
"""

from __future__ import annotations

import ipaddress
import socket
import ssl
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import certifi
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

CONNECT_TIMEOUT, READ_TIMEOUT = 5, 10
MAX_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 10
# 피싱 키트는 봇처럼 보이는 요청에 다른 화면을 보여주는 경우가 많아 일반 브라우저 UA 를 쓴다.
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/129.0 Safari/537.36")


class UnsafeDestination(Exception):
    """사설망 등 접속하면 안 되는 주소."""


def _check_public(host: str) -> None:
    """호스트가 가리키는 모든 IP 가 공인 주소인지 확인한다."""
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise ConnectionError(f"DNS 실패: {e}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise UnsafeDestination(f"{host} -> {ip}")


class _PageParser(HTMLParser):
    """<a> 태그 수와 onmouseover 속성 값만 센다."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.n_anchors = 0
        self.onmouseover: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.n_anchors += 1
        for name, value in attrs:
            if name == "onmouseover" and value:
                self.onmouseover.append(value)


def _decode(body: bytes, encoding: str | None) -> str:
    """페이지가 선언한 인코딩으로 읽되, 파이썬이 모르는 이름(예: Windows-31J)이면 UTF-8 로 읽는다."""
    try:
        return body.decode(encoding or "utf-8", errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def fetch_page(url: str, session: requests.Session) -> dict:
    """페이지를 받아 피처 계산에 필요한 정보만 돌려준다.

    반환: {"status": "ok" | "non_html" | "http_error" | "unsafe" | "dns_error" | "timeout" | "error",
           "http_status", "final_url", "n_redirects", "n_anchors", "onmouseover"}
    접속에 성공하면(본문이 HTML 이 아니어도) final_url 과 n_redirects 는 채운다.
    """
    result = {"status": "error", "http_status": None, "final_url": None,
              "n_redirects": None, "n_anchors": None, "onmouseover": None}
    current, n_redirects = url, 0
    try:
        while True:
            parts = urlsplit(current)
            if parts.scheme not in ("http", "https") or not parts.hostname:
                result["status"] = "unsafe"
                return result
            _check_public(parts.hostname)
            resp = session.get(current, allow_redirects=False, stream=True, verify=False,
                               timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            if resp.is_redirect and "location" in resp.headers:
                resp.close()
                if n_redirects >= MAX_REDIRECTS:
                    result.update(status="too_many_redirects", final_url=current, n_redirects=n_redirects)
                    return result
                current = urljoin(current, resp.headers["location"])
                n_redirects += 1
                continue
            break

        result.update(http_status=resp.status_code, final_url=current, n_redirects=n_redirects)
        if resp.status_code >= 400:
            result["status"] = "http_error"
            resp.close()
            return result
        if "text/html" not in resp.headers.get("content-type", "").lower():
            result["status"] = "non_html"
            resp.close()
            return result

        body = b""
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            body += chunk
            if len(body) >= MAX_BYTES:
                break
        resp.close()
        parser = _PageParser()
        parser.feed(_decode(body[:MAX_BYTES], resp.encoding))
        result.update(status="ok", n_anchors=parser.n_anchors, onmouseover=parser.onmouseover)
    except UnsafeDestination:
        result["status"] = "unsafe"
    except ConnectionError as e:
        result["status"] = "dns_error" if "DNS" in str(e) else "error"
    except requests.Timeout:
        result["status"] = "timeout"
    except (requests.RequestException, ValueError, UnicodeError, LookupError):
        result["status"] = "error"
    return result


def get_certificate(url: str) -> dict | None:
    """https URL 의 서버 인증서 발급사와 발급일. 검증 실패한 인증서도 정보는 읽는다.

    반환: {"issuer", "not_before", "verified", "checked_at"} 또는 None(https 아님·접속 실패)
    """
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        return None
    host, port = parts.hostname, parts.port or 443
    checked_at = datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        _check_public(host)
    except (UnsafeDestination, ConnectionError):
        return None

    def _handshake(verify: bool):
        ctx = ssl.create_default_context(cafile=certifi.where())  # 실행 환경과 무관하게 같은 CA 목록
        if not verify:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        with socket.create_connection((host, port), timeout=CONNECT_TIMEOUT) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as tls:
                return tls.getpeercert(binary_form=not verify)

    try:
        cert = _handshake(verify=True)
        issuer = dict(x[0] for x in cert.get("issuer", ()))
        return {"issuer": issuer.get("organizationName", "") or issuer.get("commonName", ""),
                "not_before": datetime.strptime(cert["notBefore"], "%b %d %H:%M:%S %Y %Z"),
                "verified": True, "checked_at": checked_at}
    except ssl.SSLCertVerificationError:
        pass  # 아래에서 검증 없이 다시 읽는다
    except (OSError, ValueError, KeyError):
        return None

    try:
        from cryptography import x509  # 검증 실패 인증서는 바이너리로 받아 직접 읽는다
        der = _handshake(verify=False)
        c = x509.load_der_x509_certificate(der)
        orgs = c.issuer.get_attributes_for_oid(x509.NameOID.ORGANIZATION_NAME)
        return {"issuer": orgs[0].value if orgs else "",
                "not_before": c.not_valid_before_utc.replace(tzinfo=None),
                "verified": False, "checked_at": checked_at}
    except Exception:
        return None


_WHOIS_CACHE: dict[str, dict | None] = {}


def lookup_whois(registered_domain: str) -> dict | None:
    """등록 도메인의 생성일·만료일. 같은 도메인은 한 번만 조회한다(캐시).

    반환: {"created", "expires", "checked_at"} 또는 None(조회 실패)
    """
    if registered_domain in _WHOIS_CACHE:
        return _WHOIS_CACHE[registered_domain]
    import whois  # python-whois

    def _first(value):
        if isinstance(value, list):
            value = value[0] if value else None
        if isinstance(value, datetime):
            return value.replace(tzinfo=None)
        return None

    result = None
    try:
        info = whois.whois(registered_domain)
        result = {"created": _first(info.creation_date), "expires": _first(info.expiration_date),
                  "checked_at": datetime.now(timezone.utc).replace(tzinfo=None)}
    except Exception:
        result = None
    _WHOIS_CACHE[registered_domain] = result
    return result
