"""v3 피처 계산 확인. 실행: python -m pytest tests/v3"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "v3"))

import features as F  # noqa: E402

PAGE = "https://shop.example.com/login"


def test_strip_query_removes_query_and_fragment_for_any_url():
    assert F.strip_query("https://a.com/x?id=1#top") == "https://a.com/x"
    assert F.strip_query("https://a.com/x") == "https://a.com/x"


def test_compute_uses_stripped_url_so_query_cannot_leak():
    with_q = F.compute("https://a.com/x?session=abc", None)
    without = F.compute("https://a.com/x", None)
    for key in ("url_len", "n_special_url", "url_entropy", "url_clean"):
        assert with_q[key] == without[key], key


def test_hosted_platform_sites_are_separate_domains():
    assert F.registered_domain("a.github.io") != F.registered_domain("b.github.io")
    assert F.url_features("https://a.github.io/")["is_hosted_platform"] == 1
    assert F.url_features("https://www.example.com/")["is_hosted_platform"] == 0


def test_bdi_values_match_paper_encoding():
    html = """<html><body>
      <img src="https://cdn.paypal.com/logo.png">
      <form action="https://evil.net/post.php"><input type="password"></form>
      <a href="/a">a</a><a href="/b">b</a><a href="https://other.org/">c</a>
    </body></html>"""
    feats, _ = F.html_features(html, PAGE)
    assert feats["bdi_logo"] == -1          # 로고가 다른 도메인
    assert feats["bdi_form_action"] == -1   # 폼이 다른 도메인으로 전송
    assert feats["bdi_common_link"] == 1    # 링크 대부분이 같은 도메인
    assert feats["suspicious_form_link"] == 1


def test_missing_elements_give_zero_not_mismatch():
    feats, _ = F.html_features("<html><body><p>hello</p></body></html>", PAGE)
    assert (feats["bdi_logo"], feats["bdi_form_action"], feats["bdi_common_link"]) == (0, 0, 0)


def test_relative_form_action_is_same_domain():
    feats, _ = F.html_features('<form action="/session"><input type="password"></form>', PAGE)
    assert feats["bdi_form_action"] == 1
    assert feats["suspicious_form_link"] == 0


def test_empty_html_is_all_zero():
    feats, texts = F.html_features("", PAGE)
    assert feats == F.EMPTY_HTML
    assert texts["title"] == texts["text"] == ""


def test_script_text_not_in_visible_text():
    html = "<html><head><title>T</title><script>var secret=1</script></head><body>Visible</body></html>"
    feats, texts = F.html_features(html, PAGE)
    assert texts["title"] == "T"
    assert "secret" not in texts["text"] and "Visible" in texts["text"]
    assert feats["n_inline_js"] == 1


def test_malformed_urls_do_not_crash():
    for url in ("https://[-randomstring1-]/a?x=1", "http://a.com:99999/", "http://a.com:abc/"):
        row = F.compute(url, "<html><a href='/x'>x</a></html>")
        assert "url_len" in row and "n_links" in row


def test_url_case_does_not_change_url_features():
    assert F.compute("https://A.com/Login/PayPal", None) == F.compute("https://a.com/login/paypal", None)


def test_trailing_slash_and_doctype_are_normalized():
    assert F.strip_query("https://A.com/") == F.strip_query("https://a.com") == "https://a.com"
    assert F.strip_query("https://a.com/x/") == "https://a.com/x"
    _, t1 = F.html_features("<!DOCTYPE html>\n<html><body>a</body></html>", PAGE)
    _, t2 = F.html_features("<html><body>a</body></html>", PAGE)
    assert t1["html_prefix"] == t2["html_prefix"]
