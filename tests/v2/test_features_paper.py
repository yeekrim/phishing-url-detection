"""features_paper.py 규칙이 논문(Phishing Websites Features) 규칙과 같은지 확인한다.

실행: python -m pytest tests/v2
"""

import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "v2"))

import features_paper as fp  # noqa: E402

L, S, P = fp.LEGIT, fp.SUSPICIOUS, fp.PHISHING
NOW = datetime(2026, 10, 11)


def test_value_scheme_matches_uci():
    assert (L, S, P) == (1, 0, -1)


def test_ip_like():
    assert fp.ip_like("http://125.98.3.123/fake.html") == P
    assert fp.ip_like("http://0x58.0xCC.0xCA.0x62/2/paypal.ca/index.html") == P  # 논문의 16진수 예시
    assert fp.ip_like("http://[2001:db8::1]/a") == P
    assert fp.ip_like("https://www.hud.ac.uk/students/") == L


def test_at_symbol():
    assert fp.have_at("http://example.com@phish.com/") == P
    assert fp.have_at("https://example.com/") == L


def test_redirection_position():
    # 논문 예시: http://www.legitimate.com//http://www.phishing.com
    assert fp.redirection("http://www.legitimate.com//http://www.phishing.com") == P
    assert fp.redirection("http://example.com/a") == L    # '//' 6번째
    assert fp.redirection("https://example.com/a") == L   # '//' 7번째


def test_tiny_url():
    assert fp.tiny_url("https://bit.ly/19DXSk4") == P
    assert fp.tiny_url("http://portal.hud.ac.uk/") == L


def test_prefix_suffix():
    assert fp.prefix_suffix("http://www.Confirme-paypal.com/") == P
    assert fp.prefix_suffix("http://www.paypal.com/a-b-c") == L  # 경로의 '-' 는 해당 없음


def test_https_rule():
    old = NOW - timedelta(days=400)
    new = NOW - timedelta(days=30)
    trusted = {"issuer": "DigiCert / GeoTrust TLS RSA CA", "verified": True, "checked_at": NOW}
    assert fp.https_rule("http://example.com/", None) == P
    assert fp.https_rule("https://example.com/", {**trusted, "not_before": old}) == L
    assert fp.https_rule("https://example.com/", {**trusted, "not_before": new}) == P   # 1년 미만
    le = {"issuer": "Let's Encrypt", "verified": True, "checked_at": NOW, "not_before": old}
    assert fp.https_rule("https://example.com/", le) == S                                # 목록에 없는 발급사
    bad = {**trusted, "not_before": old, "verified": False}
    assert fp.https_rule("https://example.com/", bad) == S                               # 검증 실패
    assert fp.https_rule("https://example.com/", None) is None                           # 인증서 못 받음


def test_domain_age_six_months():
    assert fp.domain_age(NOW - timedelta(days=200), NOW) == L
    assert fp.domain_age(NOW - timedelta(days=100), NOW) == P
    assert fp.domain_age(None, NOW) is None


def test_registration_length_one_year():
    assert fp.registration_length(NOW + timedelta(days=200), NOW) == P
    assert fp.registration_length(NOW + timedelta(days=800), NOW) == L
    assert fp.registration_length(None, NOW) is None


def test_website_forwarding():
    assert [fp.website_forwarding(n) for n in (0, 1, 2, 3, 4, 7)] == [L, L, S, S, P, P]
    assert fp.website_forwarding(None) is None


def test_status_bar_customization():
    assert fp.status_bar_customization(["window.status='http://bank.com'"]) == P
    assert fp.status_bar_customization(["this.style.color='red'"]) == L
    assert fp.status_bar_customization([]) == L
    assert fp.status_bar_customization(None) is None


def test_non_paper_features_are_raw_numbers():
    assert fp.url_depth("https://a.com/x/y/z") == 3
    assert fp.url_depth("https://a.com/") == 0
    assert fp.query_count("https://a.com/?a=1&b=2&c") == 2
    assert fp.domain_consistency("https://a.com/", "https://a.com/home") == 1
    assert fp.domain_consistency("https://a.com/", "https://b.com/") == 0
    assert fp.domain_consistency("https://a.com/", None) is None


def test_failure_never_becomes_phishing():
    """네트워크 정보가 없으면 피처는 빈 값이어야 한다 (2024년처럼 -1 로 채우지 않음)."""
    row = fp.compute("https://example.com/login", page=None, cert=None, whois=None)
    for col in ["Domain_Age", "Domain_end", "Is_Https", "Mouseover", "Web_forwards", "Hyperlinks", "Domain_Cons"]:
        assert row[col] is None, col
    assert set(fp.FEATURE_COLUMNS) <= set(row)


def test_same_url_same_features_regardless_of_label():
    """같은 입력이면 클래스와 상관없이 같은 값 (2024년 Redirection 문제 재발 방지)."""
    url = "https://secure-login.example.com/a/b?x=1"
    assert fp.compute(url, None, None, None) == fp.compute(url, None, None, None)
