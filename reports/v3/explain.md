# v3 설명 (SHAP · n-gram 가중치)

test 표본 30,000개 (정상·피싱 반반)

## SHAP — LightGBM (피처 + n-gram 점수)

방향: 피처 값이 클수록 피싱 쪽이면 +, 정상 쪽이면 -

|                     |   평균 |SHAP| |   방향(값↑→) |     정상 평균값 |    피싱 평균값 |    비중 |
|:--------------------|------------:|----------:|-----------:|----------:|------:|
| score_all           |       6.128 |     0.993 |      0.021 |     0.852 | 0.529 |
| score_url           |       1.19  |     0.982 |      0.048 |     0.828 | 0.103 |
| n_internal_links    |       0.369 |    -0.939 |    151.033 |    19.274 | 0.032 |
| html_len            |       0.312 |    -0.068 | 415804     | 92817     | 0.027 |
| host_len            |       0.229 |     0.929 |     17.155 |    23.168 | 0.02  |
| path_len            |       0.196 |    -0.71  |     33.63  |    14.493 | 0.017 |
| url_entropy         |       0.152 |     0.646 |      4.204 |     4.121 | 0.013 |
| n_external_links    |       0.135 |    -0.719 |     22.775 |     6.798 | 0.012 |
| n_links             |       0.125 |    -0.922 |    193.171 |    29.29  | 0.011 |
| self_link_ratio     |       0.119 |    -0.063 |      0.098 |     0.153 | 0.01  |
| site_label_entropy  |       0.112 |     0.712 |      2.602 |     2.858 | 0.01  |
| site_label_len      |       0.105 |     0.797 |      9.109 |    11.228 | 0.009 |
| n_hyphens_host      |       0.103 |     0.675 |      0.058 |     0.614 | 0.009 |
| host_entropy        |       0.1   |     0.56  |      3.282 |     3.614 | 0.009 |
| n_tags              |       0.095 |    -0.488 |   1450.31  |   306.116 | 0.008 |
| title_len           |       0.092 |    -0.314 |     48.514 |    29.884 | 0.008 |
| n_external_js       |       0.091 |    -0.772 |     17.011 |     3.022 | 0.008 |
| external_favicon    |       0.087 |     0.698 |      0.172 |     0.247 | 0.008 |
| has_www             |       0.087 |    -0.058 |      0.554 |     0.165 | 0.008 |
| n_css               |       0.082 |     0.786 |      8.118 |     3.462 | 0.007 |
| null_link_ratio     |       0.082 |     0.699 |      0.088 |     0.15  | 0.007 |
| external_link_ratio |       0.079 |     0.485 |      0.143 |     0.289 | 0.007 |
| n_inline_js         |       0.077 |     0.431 |     21.814 |     5.001 | 0.007 |
| n_subdomain_levels  |       0.076 |    -0.593 |      0.824 |     0.632 | 0.007 |
| n_js_src            |       0.076 |    -0.115 |     28.49  |     6.122 | 0.007 |
| url_len             |       0.074 |    -0.015 |     58.662 |    45.623 | 0.006 |
| inline_js_len       |       0.071 |    -0.581 | 126826     | 26329.8   | 0.006 |
| n_images            |       0.067 |     0.536 |     36.038 |     8.768 | 0.006 |
| is_https            |       0.06  |     0.486 |      0.98  |     0.847 | 0.005 |
| digit_ratio_url     |       0.055 |     0.356 |      0.042 |     0.063 | 0.005 |

## SHAP — LightGBM (손으로 만든 피처만)

방향: 피처 값이 클수록 피싱 쪽이면 +, 정상 쪽이면 -

|                     |   평균 |SHAP| |   방향(값↑→) |     정상 평균값 |    피싱 평균값 |    비중 |
|:--------------------|------------:|----------:|-----------:|----------:|------:|
| n_internal_links    |       0.98  |    -0.905 |    151.033 |    19.274 | 0.179 |
| path_len            |       0.549 |    -0.818 |     33.63  |    14.493 | 0.101 |
| has_www             |       0.38  |    -0.831 |      0.554 |     0.165 | 0.07  |
| path_depth          |       0.365 |    -0.906 |      2.251 |     0.77  | 0.067 |
| n_external_links    |       0.278 |    -0.792 |     22.775 |     6.798 | 0.051 |
| html_len            |       0.235 |     0.04  | 415804     | 92817     | 0.043 |
| n_links             |       0.173 |    -0.88  |    193.171 |    29.29  | 0.032 |
| n_subdomain_levels  |       0.157 |    -0.765 |      0.824 |     0.632 | 0.029 |
| bdi_common_link     |       0.135 |    -0.85  |      0.836 |     0.017 | 0.025 |
| n_js_src            |       0.124 |    -0.952 |     28.49  |     6.122 | 0.023 |
| is_https            |       0.119 |    -0.488 |      0.98  |     0.847 | 0.022 |
| external_favicon    |       0.107 |     0.705 |      0.172 |     0.247 | 0.02  |
| n_external_js       |       0.104 |    -0.865 |     17.011 |     3.022 | 0.019 |
| url_len             |       0.101 |    -0.813 |     58.662 |    45.623 | 0.018 |
| n_hyphens_host      |       0.1   |     0.673 |      0.058 |     0.614 | 0.018 |
| external_link_ratio |       0.098 |    -0.077 |      0.143 |     0.289 | 0.018 |
| self_link_ratio     |       0.084 |     0.255 |      0.098 |     0.153 | 0.015 |
| inline_js_len       |       0.079 |    -0.413 | 126826     | 26329.8   | 0.014 |
| text_len            |       0.077 |    -0.701 |  11921.2   |  2402.8   | 0.014 |
| n_iframes           |       0.07  |    -0.895 |      3.761 |     0.628 | 0.013 |
| n_images            |       0.069 |     0.895 |     36.038 |     8.768 | 0.013 |
| digit_ratio_url     |       0.068 |     0.786 |      0.042 |     0.063 | 0.012 |
| n_dots_host         |       0.059 |    -0.756 |      1.887 |     1.91  | 0.011 |
| title_len           |       0.058 |    -0.639 |     48.514 |    29.884 | 0.011 |
| host_len            |       0.056 |     0.789 |     17.155 |    23.168 | 0.01  |
| n_words             |       0.051 |    -0.644 |   1757.17  |   338.144 | 0.009 |
| host_entropy        |       0.049 |     0.726 |      3.282 |     3.614 | 0.009 |
| url_entropy         |       0.048 |     0.39  |      4.204 |     4.121 | 0.009 |
| n_digits_host       |       0.047 |     0.578 |      0.096 |     1.206 | 0.009 |
| n_digits_url        |       0.046 |     0.781 |      2.834 |     3.876 | 0.008 |

## n-gram 가중치 상위 조각 (URL+HTML+본문 선형 모델)

해시 충돌로 한 번호에 여러 조각이 들어갈 수 있다 (대응표: 학습 파일 4개에서 수집)

### n-gram 블록: url

| 방향   |    가중치 | 조각                  |
|:-----|-------:|:--------------------|
| 피싱 쪽 |  4.57  | .top                |
| 피싱 쪽 |  4.026 | php / ra/x2         |
| 피싱 쪽 |  3.871 | .we / ee.ta / rroco |
| 피싱 쪽 |  3.87  | url                 |
| 피싱 쪽 |  3.76  | .php                |
| 피싱 쪽 |  3.563 | .shop               |
| 피싱 쪽 |  3.556 | inhoo / flow.       |
| 피싱 쪽 |  3.547 | site                |
| 피싱 쪽 |  3.519 | .to                 |
| 피싱 쪽 |  3.442 | .ph                 |
| 피싱 쪽 |  3.358 | http:               |
| 피싱 쪽 |  3.354 | mwal / tp:/ / 56.r  |
| 피싱 쪽 |  3.313 | .web                |
| 피싱 쪽 |  3.291 | ttp:                |
| 피싱 쪽 |  3.266 | tp: / mm_ / xjxmy   |
| 피싱 쪽 |  3.244 | p:/ / cgmro         |
| 피싱 쪽 |  3.225 | ttp:/               |
| 피싱 쪽 |  3.224 | p://                |
| 피싱 쪽 |  3.199 | am-a / .sho         |
| 피싱 쪽 |  3.174 | ldpla / tp://       |
| 피싱 쪽 |  3.145 | .dev                |
| 피싱 쪽 |  3.121 | .cn / 22-ta / vada. |
| 피싱 쪽 |  2.983 | ://                 |
| 피싱 쪽 |  2.965 | htt                 |
| 피싱 쪽 |  2.965 | ttp                 |
| 정상 쪽 | -7.484 | .jp                 |
| 정상 쪽 | -6.497 | jp/ / lagen / ryana |
| 정상 쪽 | -6.15  | .jp/ / /grie        |
| 정상 쪽 | -5.898 | .go / fu8s/         |
| 정상 쪽 | -5.052 | .gov                |
| 정상 쪽 | -3.561 | .pa. / ec3- / .edu  |
| 정상 쪽 | -3.551 | us/                 |
| 정상 쪽 | -3.49  | .ed / deuta         |
| 정상 쪽 | -3.489 | gov                 |
| 정상 쪽 | -3.481 | ww.                 |
| 정상 쪽 | -3.474 | www.                |
| 정상 쪽 | -3.449 | www                 |
| 정상 쪽 | -3.43  | .it                 |
| 정상 쪽 | -3.386 | .com/               |
| 정상 쪽 | -3.37  | es/ / ngi.c         |
| 정상 쪽 | -3.272 | com/                |
| 정상 쪽 | -3.258 | /en                 |
| 정상 쪽 | -3.252 | .au                 |
| 정상 쪽 | -3.233 | .nl / 9np / a4.v    |
| 정상 쪽 | -3.198 | om/                 |
| 정상 쪽 | -3.092 | /20                 |
| 정상 쪽 | -3.079 | ts/ / rmspo / etmsk |
| 정상 쪽 | -3.063 | et/                 |
| 정상 쪽 | -3.022 | er/ / itli / cur.c  |
| 정상 쪽 | -3.012 | .ca                 |

### n-gram 블록: html

| 방향   |    가중치 | 조각                                                                         |
|:-----|-------:|:---------------------------------------------------------------------------|
| 피싱 쪽 | 10.052 | body html / hebrew speak                                                   |
| 피싱 쪽 |  5.825 | en head / checkyourmath com / ficon fe8wk9rgiuiokfgdv0kty                  |
| 피싱 쪽 |  5.391 | head body / stock stocks                                                   |
| 피싱 쪽 |  4.908 | div body                                                                   |
| 피싱 쪽 |  4.882 | script body / ffffff color_header_subnav_border                            |
| 피싱 쪽 |  3.707 | france up / robinhood / id dejam_tag                                       |
| 피싱 쪽 |  3.671 | php / fbps1601 html / hsw assethost                                        |
| 피싱 쪽 |  3.489 | nofollow / qgdqugjmtqwypg4yzxlci26dbqliv7u fsirfxhz2niavyd17m2d5fsh8ux15pg |
| 피싱 쪽 |  3.387 | footer / bacio meta / hisc673 clearcareonline                              |
| 피싱 쪽 |  3.238 | lang / var buttonsrc / framem document                                     |
| 피싱 쪽 |  3.077 | css head / fbc0isq0fddeisaujbqw9clenbq3bcl                                 |
| 피싱 쪽 |  2.982 | body div / ohom and / content gis                                          |
| 피싱 쪽 |  2.885 | cpbslbuy / comments / ee8yf3mnudfta0bvata2                                 |
| 피싱 쪽 |  2.865 | script / common a5ce2b1cceb139bc2435                                       |
| 피싱 쪽 |  2.82  | co / us-en meta / cevent811 amp                                            |
| 피싱 쪽 |  2.815 | html / s-ntf1-cdo30d10 s-wpo-pr1-dysize                                    |
| 피싱 쪽 |  2.757 | unescape / node--type-dynamic-page href                                    |
| 피싱 쪽 |  2.639 | email / observe function                                                   |
| 피싱 쪽 |  2.606 | head                                                                       |
| 피싱 쪽 |  2.595 | calif / en-us head / wsite-title wsite-not-footer                          |
| 피싱 쪽 |  2.541 | base64                                                                     |
| 피싱 쪽 |  2.506 | prev_selected / argentina https / noindex nofollow                         |
| 피싱 쪽 |  2.496 | lang pt-br / fine jewelry / ykbzkb4uhrspty e2hpg5qmo                       |
| 피싱 쪽 |  2.456 | required / t12 datemodif / antivirus description                           |
| 피싱 쪽 |  2.434 | html style / page kia-research                                             |
| 정상 쪽 | -5.612 | html lang                                                                  |
| 정상 쪽 | -3.676 | bitter / href favicon / fopen-file-folder_1f4c2 png                        |
| 정상 쪽 | -3.094 | blank / next feb15d14-e3e6-4c6a-9d51-3569237                               |
| 정상 쪽 | -3.075 | text css / basilisk harry / following heated                               |
| 정상 쪽 | -3.051 | javascript src / highbandwidth data-di-loaded                              |
| 정상 쪽 | -3.037 | href http / googleplus / territories meta                                  |
| 정상 쪽 | -2.866 | ar web / head title / hdg-green border-radius                              |
| 정상 쪽 | -2.843 | stylesheet link / wzmcgdtaaacbaaabpkaaamrigp4o2dh                          |
| 정상 쪽 | -2.707 | io-p-500 / kpthrive meta / name description                                |
| 정상 쪽 | -2.595 | the / xt en / href cdn-4                                                   |
| 정상 쪽 | -2.589 | au ca / target blank                                                       |
| 정상 쪽 | -2.476 | async src / libs enquire                                                   |
| 정상 쪽 | -2.449 | head style / video-box video / px w3m-radius-button                        |
| 정상 쪽 | -2.437 | text javascript                                                            |
| 정상 쪽 | -2.424 | images / altfind cookiestr / block o-object-fit                            |
| 정상 쪽 | -2.377 | lib / oxygenation / grid-nav-item-001 click                                |
| 정상 쪽 | -2.311 | http / hina / auto-publish content                                         |
| 정상 쪽 | -2.298 | ja / volcker_paul / jsname r4nke                                           |
| 정상 쪽 | -2.294 | did prove / lander script                                                  |
| 정상 쪽 | -2.292 | onload / kykqysntejel0aq9oqdcsgg                                           |
| 정상 쪽 | -2.257 | display none / located time / bf5a3f8d611ff661cad6                         |
| 정상 쪽 | -2.257 | table / ui css                                                             |
| 정상 쪽 | -2.219 | lang ja / ea97c vert-700 / id rate-my-post-css                             |
| 정상 쪽 | -2.195 | content                                                                    |
| 정상 쪽 | -2.18  | href content / jquery transit                                              |

### n-gram 블록: text

| 방향   |    가중치 | 조각                                                   |
|:-----|-------:|:-----------------------------------------------------|
| 피싱 쪽 |  6.731 | webmail / star pauline                               |
| 피싱 쪽 |  5.665 | 講談社 / modelista / robinhood                          |
| 피싱 쪽 |  5.517 | whatsapp                                             |
| 피싱 쪽 |  5.495 | meta / berlin upper                                  |
| 피싱 쪽 |  5.337 | wallet / by wapo / move aimed                        |
| 피싱 쪽 |  4.617 | sfr / here to / amaris is                            |
| 피싱 쪽 |  4.446 | mail / ambulance ambulatory / neurogenic orthostatic |
| 피싱 쪽 |  4.406 | facebook / 各種設定方法 ブラウザの設定                            |
| 피싱 쪽 |  4.355 | security / storage section                           |
| 피싱 쪽 |  4.088 | happen by / incorrect / fundsroute                   |
| 피싱 쪽 |  3.944 | bt / 2024年8月23日                                      |
| 피싱 쪽 |  3.846 | crypto / and break / current rank                    |
| 피싱 쪽 |  3.718 | de / ken is / purple red                             |
| 피싱 쪽 |  3.714 | verify / altresì fatto / choices check               |
| 피싱 쪽 |  3.638 | abuse / turnover in                                  |
| 피싱 쪽 |  3.616 | banking / scsi sas / blenders food                   |
| 피싱 쪽 |  3.339 | trading / often is / marble ceramic                  |
| 피싱 쪽 |  3.309 | secure / initiativesbuying rolexcare                 |
| 피싱 쪽 |  3.28  | bank / april 26th / developmentsbecause they         |
| 피싱 쪽 |  3.185 | dhl                                                  |
| 피싱 쪽 |  3.132 | telegram / halifax nova / actieffunctionele          |
| 피싱 쪽 |  3.078 | in smartwatch / webmail login                        |
| 피싱 쪽 |  3.04  | para / 662big ass1                                   |
| 피싱 쪽 |  2.937 | agoge / confirm / tein street                        |
| 피싱 쪽 |  2.92  | home home / よみぽ1ポイントプレゼント                            |
| 정상 쪽 | -3.571 | search                                               |
| 정상 쪽 | -3.236 | pieces / to the / 00 住信sbiネット銀行アプリ                   |
| 정상 쪽 | -3.111 | staff / used bmw / rare purple                       |
| 정상 쪽 | -2.819 | boiled / resources                                   |
| 정상 쪽 | -2.677 | county / towels ceramic / ブラックサンダーは1994年に発売されました 当時  |
| 정상 쪽 | -2.674 | grm / tinned / version                               |
| 정상 쪽 | -2.569 | reset / همکار 25                                     |
| 정상 쪽 | -2.529 | llc / お知らせ一覧 アレルギー成分について                             |
| 정상 쪽 | -2.51  | time / are considering / dɪˈspaɪzɪz dɪˈspaɪzɪz       |
| 정상 쪽 | -2.491 | forgot / you physical                                |
| 정상 쪽 | -2.489 | close                                                |
| 정상 쪽 | -2.447 | health                                               |
| 정상 쪽 | -2.445 | inc / 油彩額縁                                           |
| 정상 쪽 | -2.426 | events / are flickr / cityknoxvillelas               |
| 정상 쪽 | -2.405 | calendar / disponible de / parking hourly            |
| 정상 쪽 | -2.374 | family / and fab / unificationmulti table            |
| 정상 쪽 | -2.363 | porn / 経済 グルメ / in fusion                            |
| 정상 쪽 | -2.351 | for the / 21 winter / view excel                     |
| 정상 쪽 | -2.323 | years / year age / מזמור שרתה                        |
| 정상 쪽 | -2.293 | join / laws or / mail 2025                           |
| 정상 쪽 | -2.282 | portal / registre una                                |
| 정상 쪽 | -2.256 | 55 43 / logging / love got                           |
| 정상 쪽 | -2.251 | current                                              |
| 정상 쪽 | -2.245 | sso / 10 251st                                       |
| 정상 쪽 | -2.234 | history / medical policies / uhfffaoysa computed     |
