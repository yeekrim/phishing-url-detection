# v3 설명 (SHAP · n-gram 가중치)

test 표본 30,000개 (정상·피싱 반반)

## SHAP — LightGBM (피처 + n-gram 점수)

방향: 피처 값이 클수록 피싱 쪽이면 +, 정상 쪽이면 -

|                     |   평균 |SHAP| |   방향(값↑→) |     정상 평균값 |    피싱 평균값 |    비중 |
|:--------------------|------------:|----------:|-----------:|----------:|------:|
| score_all           |       4.869 |     0.994 |      0.028 |     0.826 | 0.598 |
| score_url           |       0.895 |     0.95  |      0.061 |     0.831 | 0.11  |
| n_internal_links    |       0.244 |    -0.912 |    151.033 |    19.274 | 0.03  |
| html_len            |       0.159 |    -0.02  | 415804     | 92817     | 0.02  |
| host_len            |       0.147 |     0.916 |     17.155 |    23.168 | 0.018 |
| path_len            |       0.13  |    -0.67  |     33.63  |    14.493 | 0.016 |
| url_entropy         |       0.109 |     0.711 |      4.204 |     4.121 | 0.013 |
| self_link_ratio     |       0.092 |     0.211 |      0.098 |     0.153 | 0.011 |
| n_external_links    |       0.071 |    -0.692 |     22.775 |     6.798 | 0.009 |
| has_www             |       0.068 |     0.779 |      0.554 |     0.165 | 0.008 |
| external_favicon    |       0.065 |     0.705 |      0.172 |     0.247 | 0.008 |
| host_entropy        |       0.064 |     0.674 |      3.282 |     3.614 | 0.008 |
| n_hyphens_host      |       0.064 |     0.65  |      0.058 |     0.614 | 0.008 |
| n_links             |       0.061 |    -0.909 |    193.171 |    29.29  | 0.007 |
| n_inline_js         |       0.06  |     0.661 |     21.814 |     5.001 | 0.007 |
| title_len           |       0.054 |    -0.272 |     48.514 |    29.884 | 0.007 |
| url_len             |       0.05  |     0.419 |     58.662 |    45.623 | 0.006 |
| longest_host_token  |       0.048 |     0.76  |      9.156 |    10.547 | 0.006 |
| n_css               |       0.047 |     0.767 |      8.118 |     3.462 | 0.006 |
| external_link_ratio |       0.046 |     0.355 |      0.143 |     0.289 | 0.006 |
| n_js_src            |       0.045 |    -0.425 |     28.49  |     6.122 | 0.006 |
| n_subdomain_levels  |       0.044 |    -0.624 |      0.824 |     0.632 | 0.005 |
| n_iframes           |       0.044 |     0.743 |      3.761 |     0.628 | 0.005 |
| is_https            |       0.039 |     0.484 |      0.98  |     0.847 | 0.005 |
| text_len            |       0.038 |    -0.181 |  11921.2   |  2402.8   | 0.005 |
| n_tags              |       0.036 |    -0.226 |   1450.31  |   306.116 | 0.004 |
| inline_js_len       |       0.034 |    -0.425 | 126826     | 26329.8   | 0.004 |
| digit_ratio_url     |       0.033 |     0.608 |      0.042 |     0.063 | 0.004 |
| n_images            |       0.032 |     0.215 |     36.038 |     8.768 | 0.004 |
| n_external_js       |       0.031 |    -0.464 |     17.011 |     3.022 | 0.004 |

## SHAP — LightGBM (손으로 만든 피처만)

방향: 피처 값이 클수록 피싱 쪽이면 +, 정상 쪽이면 -

|                     |   평균 |SHAP| |   방향(값↑→) |     정상 평균값 |    피싱 평균값 |    비중 |
|:--------------------|------------:|----------:|-----------:|----------:|------:|
| n_internal_links    |       1.577 |    -0.945 |    151.033 |    19.274 | 0.146 |
| path_len            |       1.025 |    -0.917 |     33.63  |    14.493 | 0.095 |
| has_www             |       0.66  |    -0.831 |      0.554 |     0.165 | 0.061 |
| html_len            |       0.584 |     0.018 | 415804     | 92817     | 0.054 |
| is_hosted_platform  |       0.531 |     0.494 |      0.002 |     0.177 | 0.049 |
| path_depth          |       0.428 |    -0.907 |      2.251 |     0.77  | 0.04  |
| n_external_links    |       0.345 |    -0.814 |     22.775 |     6.798 | 0.032 |
| n_external_js       |       0.295 |    -0.875 |     17.011 |     3.022 | 0.027 |
| n_subdomain_levels  |       0.29  |    -0.737 |      0.824 |     0.632 | 0.027 |
| is_https            |       0.281 |    -0.488 |      0.98  |     0.847 | 0.026 |
| n_words             |       0.253 |    -0.665 |   1757.17  |   338.144 | 0.023 |
| url_entropy         |       0.228 |     0.507 |      4.204 |     4.121 | 0.021 |
| n_js_src            |       0.227 |    -0.925 |     28.49  |     6.122 | 0.021 |
| n_hyphens_host      |       0.22  |     0.676 |      0.058 |     0.614 | 0.02  |
| external_favicon    |       0.209 |     0.705 |      0.172 |     0.247 | 0.019 |
| host_len            |       0.175 |     0.852 |     17.155 |    23.168 | 0.016 |
| n_links             |       0.174 |    -0.939 |    193.171 |    29.29  | 0.016 |
| title_len           |       0.161 |    -0.413 |     48.514 |    29.884 | 0.015 |
| n_images            |       0.159 |     0.831 |     36.038 |     8.768 | 0.015 |
| inline_js_len       |       0.147 |    -0.683 | 126826     | 26329.8   | 0.014 |
| self_link_ratio     |       0.145 |    -0.35  |      0.098 |     0.153 | 0.013 |
| n_iframes           |       0.138 |    -0.86  |      3.761 |     0.628 | 0.013 |
| n_digits_host       |       0.135 |     0.591 |      0.096 |     1.206 | 0.013 |
| n_inline_js         |       0.126 |    -0.516 |     21.814 |     5.001 | 0.012 |
| n_css               |       0.123 |     0.473 |      8.118 |     3.462 | 0.011 |
| n_suspicious_words  |       0.122 |     0.462 |      0.07  |     0.105 | 0.011 |
| external_link_ratio |       0.116 |     0.373 |      0.143 |     0.289 | 0.011 |
| n_tags              |       0.115 |     0.093 |   1450.31  |   306.116 | 0.011 |
| digit_ratio_url     |       0.115 |     0.711 |      0.042 |     0.063 | 0.011 |
| bdi_common_link     |       0.112 |    -0.856 |      0.836 |     0.017 | 0.01  |

## n-gram 가중치 상위 조각 (URL+HTML+본문 선형 모델)

해시 충돌로 한 번호에 여러 조각이 들어갈 수 있다 (대응표: 학습 파일 4개에서 수집)

### n-gram 블록: url

| 방향   |    가중치 | 조각                  |
|:-----|-------:|:--------------------|
| 피싱 쪽 |  3.481 | .top                |
| 피싱 쪽 |  3.404 | .we / rroco / ee.ta |
| 피싱 쪽 |  3.383 | php / ra/x2         |
| 피싱 쪽 |  3.246 | .php                |
| 피싱 쪽 |  3.099 | inhoo / flow.       |
| 피싱 쪽 |  3.068 | site                |
| 피싱 쪽 |  2.974 | .shop               |
| 피싱 쪽 |  2.97  | 56.r / tp:/ / mwal  |
| 피싱 쪽 |  2.944 | http:               |
| 피싱 쪽 |  2.92  | p:/ / cgmro         |
| 피싱 쪽 |  2.919 | .ph                 |
| 피싱 쪽 |  2.918 | ttp:                |
| 피싱 쪽 |  2.916 | p://                |
| 피싱 쪽 |  2.914 | ttp:/               |
| 피싱 쪽 |  2.901 | mm_ / tp: / xjxmy   |
| 피싱 쪽 |  2.898 | .to                 |
| 피싱 쪽 |  2.883 | tp:// / ldpla       |
| 피싱 쪽 |  2.88  | url                 |
| 피싱 쪽 |  2.815 | am-a / .sho         |
| 피싱 쪽 |  2.736 | ://                 |
| 피싱 쪽 |  2.729 | htt                 |
| 피싱 쪽 |  2.729 | hina / http         |
| 피싱 쪽 |  2.729 | ttp                 |
| 피싱 쪽 |  2.658 | .web                |
| 피싱 쪽 |  2.638 | .dev                |
| 정상 쪽 | -6.194 | .jp                 |
| 정상 쪽 | -5.305 | jp/ / lagen / ryana |
| 정상 쪽 | -5.079 | .jp/ / /grie        |
| 정상 쪽 | -4.266 | .go / fu8s/         |
| 정상 쪽 | -3.879 | .gov                |
| 정상 쪽 | -3.287 | .com/               |
| 정상 쪽 | -3.233 | com/                |
| 정상 쪽 | -3.227 | om/                 |
| 정상 쪽 | -3.021 | es/ / ngi.c         |
| 정상 쪽 | -2.975 | gov                 |
| 정상 쪽 | -2.853 | /en                 |
| 정상 쪽 | -2.85  | ec3- / .pa. / .edu  |
| 정상 쪽 | -2.817 | .ed / deuta         |
| 정상 쪽 | -2.786 | us/                 |
| 정상 쪽 | -2.778 | .or                 |
| 정상 쪽 | -2.765 | org / izs. / dget.  |
| 정상 쪽 | -2.718 | .it                 |
| 정상 쪽 | -2.651 | .org / 11785        |
| 정상 쪽 | -2.647 | .au                 |
| 정상 쪽 | -2.645 | ww.                 |
| 정상 쪽 | -2.643 | et/                 |
| 정상 쪽 | -2.631 | www.                |
| 정상 쪽 | -2.482 | it/                 |
| 정상 쪽 | -2.476 | ts/ / rmspo / etmsk |
| 정상 쪽 | -2.47  | www                 |

### n-gram 블록: html

| 방향   |    가중치 | 조각                                                                         |
|:-----|-------:|:---------------------------------------------------------------------------|
| 피싱 쪽 |  7.614 | body html / hebrew speak                                                   |
| 피싱 쪽 |  4.919 | en head / checkyourmath com / ficon fe8wk9rgiuiokfgdv0kty                  |
| 피싱 쪽 |  4.735 | head body / stock stocks                                                   |
| 피싱 쪽 |  3.314 | script body / ffffff color_header_subnav_border                            |
| 피싱 쪽 |  3.073 | div body                                                                   |
| 피싱 쪽 |  2.817 | france up / robinhood / id dejam_tag                                       |
| 피싱 쪽 |  2.74  | nofollow / qgdqugjmtqwypg4yzxlci26dbqliv7u fsirfxhz2niavyd17m2d5fsh8ux15pg |
| 피싱 쪽 |  2.734 | php / fbps1601 html / hsw assethost                                        |
| 피싱 쪽 |  2.332 | script / common a5ce2b1cceb139bc2435                                       |
| 피싱 쪽 |  2.276 | body / js afzewbpod / showwatermark                                        |
| 피싱 쪽 |  2.261 | css head / fbc0isq0fddeisaujbqw9clenbq3bcl                                 |
| 피싱 쪽 |  2.23  | html / s-ntf1-cdo30d10 s-wpo-pr1-dysize                                    |
| 피싱 쪽 |  2.2   | footer / bacio meta / hisc673 clearcareonline                              |
| 피싱 쪽 |  2.183 | required / t12 datemodif / antivirus description                           |
| 피싱 쪽 |  2.167 | ohom and / body div / content gis                                          |
| 피싱 쪽 |  2.129 | head                                                                       |
| 피싱 쪽 |  2.071 | base64                                                                     |
| 피싱 쪽 |  2.065 | unescape / node--type-dynamic-page href                                    |
| 피싱 쪽 |  2.06  | cpbslbuy / comments / ee8yf3mnudfta0bvata2                                 |
| 피싱 쪽 |  1.979 | account / t02 breadcrumb / personal_storage_box                            |
| 피싱 쪽 |  1.977 | prev_selected / argentina https / noindex nofollow                         |
| 피싱 쪽 |  1.922 | email / observe function                                                   |
| 피싱 쪽 |  1.884 | lang / var buttonsrc / framem document                                     |
| 피싱 쪽 |  1.861 | rel icon / ayob2hvpm0p4zmup / rotor-icon-circle rotor-icon-clear           |
| 피싱 쪽 |  1.832 | icon / content melinda / windows-phone bitesize                            |
| 정상 쪽 | -3.935 | html lang                                                                  |
| 정상 쪽 | -2.62  | bitter / href favicon / fopen-file-folder_1f4c2 png                        |
| 정상 쪽 | -2.484 | text css / basilisk harry / following heated                               |
| 정상 쪽 | -2.44  | javascript src / highbandwidth data-di-loaded                              |
| 정상 쪽 | -2.255 | blank / next feb15d14-e3e6-4c6a-9d51-3569237                               |
| 정상 쪽 | -2.247 | async src / libs enquire                                                   |
| 정상 쪽 | -2.212 | href http / googleplus / territories meta                                  |
| 정상 쪽 | -2.162 | ar web / head title / hdg-green border-radius                              |
| 정상 쪽 | -2.145 | the / xt en / href cdn-4                                                   |
| 정상 쪽 | -2.036 | io-p-500 / kpthrive meta / name description                                |
| 정상 쪽 | -1.992 | stylesheet link / wzmcgdtaaacbaaabpkaaamrigp4o2dh                          |
| 정상 쪽 | -1.975 | head style / video-box video / px w3m-radius-button                        |
| 정상 쪽 | -1.961 | text javascript                                                            |
| 정상 쪽 | -1.879 | au ca / target blank                                                       |
| 정상 쪽 | -1.874 | onload / kykqysntejel0aq9oqdcsgg                                           |
| 정상 쪽 | -1.838 | base / v45 bst / hsun2g6khlvm6dm-vbt4lv7xzpxcfml                           |
| 정상 쪽 | -1.825 | images / altfind cookiestr / block o-object-fit                            |
| 정상 쪽 | -1.823 | hina / http / auto-publish content                                         |
| 정상 쪽 | -1.787 | table / ui css                                                             |
| 정상 쪽 | -1.764 | googletagmanager com / tawxyfh6vd66pfkbzhxcosdozyzqe6c                     |
| 정상 쪽 | -1.759 | themes                                                                     |
| 정상 쪽 | -1.749 | hotel or / meta conte / googletagmanager                                   |
| 정상 쪽 | -1.735 | js id / tout au / in margin-bottom                                         |
| 정상 쪽 | -1.729 | www googletagmanager                                                       |
| 정상 쪽 | -1.698 | article                                                                    |

### n-gram 블록: text

| 방향   |    가중치 | 조각                                                   |
|:-----|-------:|:-----------------------------------------------------|
| 피싱 쪽 |  5.172 | 講談社 / modelista / robinhood                          |
| 피싱 쪽 |  5.034 | webmail / star pauline                               |
| 피싱 쪽 |  4.355 | meta / berlin upper                                  |
| 피싱 쪽 |  4.084 | wallet / by wapo / move aimed                        |
| 피싱 쪽 |  3.814 | mail / ambulance ambulatory / neurogenic orthostatic |
| 피싱 쪽 |  3.814 | facebook / 各種設定方法 ブラウザの設定                            |
| 피싱 쪽 |  3.708 | sfr / here to / amaris is                            |
| 피싱 쪽 |  3.665 | whatsapp                                             |
| 피싱 쪽 |  3.167 | de / ken is / purple red                             |
| 피싱 쪽 |  3.136 | security / storage section                           |
| 피싱 쪽 |  3.119 | verify / altresì fatto / choices check               |
| 피싱 쪽 |  2.987 | crypto / and break / current rank                    |
| 피싱 쪽 |  2.81  | bt / 2024年8月23日                                      |
| 피싱 쪽 |  2.768 | abuse / turnover in                                  |
| 피싱 쪽 |  2.712 | para / 662big ass1                                   |
| 피싱 쪽 |  2.704 | happen by / incorrect / fundsroute                   |
| 피싱 쪽 |  2.679 | banking / scsi sas / blenders food                   |
| 피싱 쪽 |  2.586 | trading / often is / marble ceramic                  |
| 피싱 쪽 |  2.523 | dhl                                                  |
| 피싱 쪽 |  2.463 | secure / initiativesbuying rolexcare                 |
| 피싱 쪽 |  2.33  | bank / april 26th / developmentsbecause they         |
| 피싱 쪽 |  2.311 | home home / よみぽ1ポイントプレゼント                            |
| 피싱 쪽 |  2.306 | agoge / confirm / tein street                        |
| 피싱 쪽 |  2.285 | webmail login / in smartwatch                        |
| 피싱 쪽 |  2.266 | xls usalo / verification / hannah madden             |
| 정상 쪽 | -2.585 | to the / pieces / 00 住信sbiネット銀行アプリ                   |
| 정상 쪽 | -2.488 | search                                               |
| 정상 쪽 | -2.454 | boiled / resources                                   |
| 정상 쪽 | -2.253 | an / mi ve                                           |
| 정상 쪽 | -2.221 | forgot / you physical                                |
| 정상 쪽 | -2.182 | health                                               |
| 정상 쪽 | -2.153 | county / towels ceramic / ブラックサンダーは1994年に発売されました 当時  |
| 정상 쪽 | -2.15  | calendar / disponible de / parking hourly            |
| 정상 쪽 | -2.095 | reset / همکار 25                                     |
| 정상 쪽 | -2.063 | skip / haier 3kg縦型洗濯機 / disposables available        |
| 정상 쪽 | -2.06  | close                                                |
| 정상 쪽 | -2.059 | main / dpfs are / about 1506                         |
| 정상 쪽 | -1.989 | inc / 油彩額縁                                           |
| 정상 쪽 | -1.965 | events / are flickr / cityknoxvillelas               |
| 정상 쪽 | -1.877 | porn / 経済 グルメ / in fusion                            |
| 정상 쪽 | -1.874 | system / views 720pthis / 08066633646 通話料無料          |
| 정상 쪽 | -1.869 | all / still perhaps / complémentaires                |
| 정상 쪽 | -1.854 | del                                                  |
| 정상 쪽 | -1.85  | news / free now / metamask loginthe                  |
| 정상 쪽 | -1.846 | 99 general / questo gioco / your password            |
| 정상 쪽 | -1.841 | of the / sets baking / 中断証明書をお持ちの方 現在の契約が共済の方        |
| 정상 쪽 | -1.819 | faq / felt relief / col together                     |
| 정상 쪽 | -1.817 | family / and fab / unificationmulti table            |
| 정상 쪽 | -1.816 | years / year age / מזמור שרתה                        |
| 정상 쪽 | -1.814 | state / producttype / rating global                  |
