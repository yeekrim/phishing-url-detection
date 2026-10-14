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


def test_site_label_is_name_under_suffix():
    hosted = F.url_features("https://secure-login1.netlify.app")
    assert (hosted["site_label_len"], hosted["site_label_digits"], hosted["site_label_hyphens"]) == (13, 1, 1)
    assert hosted["site_label_suspicious"] == 1
    assert F.url_features("https://www.paypal.com")["site_label_len"] == 6
    assert F.url_features("http://1.2.3.4/a")["site_label_len"] == 0


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


def test_label_groups_follow_fixed_rules():
    from label_groups import label_group
    assert label_group("google", "https://abc.blogspot.com.au") == "blogspot_google"
    assert label_group("google", "https://accounts-google.example.com/x") == "other"
    assert label_group("facebook", "https://abc.blogspot.com") == "other"
    assert label_group("GrandLisboaMacau", "https://1.2.3.4:4777") == "casino"
    assert label_group("bet365", "https://b03.co") == "betting"
    assert label_group("jansport", "https://x.com") == "other"


def test_brand_domain_match_cases():
    def b(url, title, brands=frozenset()):
        return F.brand_features(title, "", url, brands)
    assert b("https://www.correios.com.br/login", "Correios")["title_domain_match"] == 1
    assert b("https://banco.bradesco", "Banco Bradesco | Você primeiro")["title_domain_match"] == 1
    assert b("https://sbi.co.in", "State Bank of India")["title_domain_match"] == 1      # 머리글자
    fake = b("https://rricrosoft-office.org/i/x", "Microsoft")
    assert fake["title_domain_match"] == 0 and fake["title_lookalike"] == 1
    generic = b("https://login.ionos.de", "Login – Zugang zu Ihrem Kundenkonto")      # 흔한 단어끼리는 무시
    assert generic["title_domain_match"] == 0 and generic["title_lookalike"] == 0
    assert b("https://paypal.secure-x.com/a", "PayPal")["title_domain_match"] == 0      # 하위 도메인의 브랜드는 일치 아님


def test_known_brand_mismatch_and_mentions():
    brands = frozenset({"paypal", "microsoft"})
    assert F.brand_features("PayPal: Log in", "", "https://evil.top", brands)["known_brand_mismatch"] == 1
    assert F.brand_features("PayPal: Log in", "", "https://www.paypal.com", brands)["known_brand_mismatch"] == 0
    assert F.brand_features("Hi", "Correios ... correios.com.br", "https://correios.com.br", brands)["text_domain_mentions"] == 2
    assert F.brand_features("", "", "http://1.2.3.4/a", brands) == {
        "title_domain_match": 0, "title_domain_sim": 0.0, "title_lookalike": 0,
        "text_domain_mentions": 0, "known_brand_mismatch": 0}


def test_platforms_missing_from_bundled_psl_are_hosted():
    for host in ("a.blogspot.cl", "a.blogspot.com.au", "x.gitbook.io", "shop.weebly.com", "a.b.wordpress.com"):
        assert F.url_features(f"https://{host}")["is_hosted_platform"] == 1, host
    assert F.registered_domain("a.blogspot.cl") != F.registered_domain("b.blogspot.cl")
    assert F.registered_domain("a.b.wordpress.com") == "b.wordpress.com"
    assert F.url_features("https://a.blogspot.cl")["site_label_len"] == 1
    for host in ("weebly.com", "www.weebly.com", "a.kleinanzegien.de", "a.blogspot.com"):
        expected = int(host == "a.blogspot.com")   # blogspot.com 은 원래 사설 접미사 목록에 있음
        assert F.url_features(f"https://{host}")["is_hosted_platform"] == expected, host


def test_mask_jp_only_changes_jp_suffix():
    assert F.mask_jp("https://www.example.co.jp/login") == "https://www.example.com/login"
    assert F.mask_jp("https://shop.jp") == "https://shop.com"
    assert F.mask_jp("https://city.tokyo.jp/a") == "https://city.com/a"
    assert F.mask_jp("https://jp.example.com/jp") == "https://jp.example.com/jp"
    assert F.mask_jp("http://1.2.3.4/x.jp") == "http://1.2.3.4/x.jp"
