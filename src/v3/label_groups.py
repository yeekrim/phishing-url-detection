"""라벨 범위 점검용 묶음 규칙. (v3 한계 1번)

최종 모델이 놓친 피싱의 약 43% 가 아래 묶음에 몰려 있었다(오탐률 0.1% 기준).
이 묶음이 "브랜드를 사칭해 자격증명을 빼가는 피싱" 과 성격이 다른지 확인하려고 묶는다.

규칙은 평가 점수를 보기 전에 PhreshPhish 의 target 이름과 URL 만으로 정한다.
점수가 오르는 쪽으로 묶음을 고르지 않기 위해서다. 공식 점수는 이 묶음을 빼지 않은 값이다.

    blogspot_google : 호스트가 *.blogspot.* 이고 target 이 google
                      (test 에서 놓친 blogspot 표본이 평범한 블로그 제목이었다)
    casino          : 카지노 리조트 브랜드 target
    betting         : 온라인 도박·스포츠 베팅 브랜드 target

betting·casino 목록은 train+test 의 target 이름 3,616개 중 bet·casino·macau·win·sport 등이 들어간 이름을
하나씩 보고 도박 브랜드만 골랐다. 스포츠 용품(jansport, brooks sports), windows, twint 등은 넣지 않았다.
playon sports 처럼 무엇인지 확실하지 않은 이름도 넣지 않았다.
"""

from __future__ import annotations

import re

CASINO = {
    "grandlisboamacau", "mgm_macau", "wynn_resorts_macau", "crowncasinomelbourne", "sandscasino",
    "motorcity casino", "naskila casino", "casinometropol", "888casino", "zodiac casino",
}
BETTING = {
    "betwinner", "bet365", "jojobet", "holiganbet", "sahabet", "sbotop", "bets10", "casino rewards", "baywin",
    "onwin", "matadorbet", "sun win", "sun_win", "mariobet", "b1bet", "tipobet365", "betbra", "meritbet",
    "linebet", "extrabet", "madridbet", "brbet", "fixbet", "stake", "stakebet", "bettilt", "bwin", "betist",
    "sportybet", "fullbet", "fulltbet", "tarafbet", "starzbet", "sportsbet", "sekabet", "betespecial", "betou",
    "bycasino", "casino", "casino action", "casinomaxi", "dafabet", "sbobet", "estrelabet", "tsb gaming ltd",
}
BLOGSPOT = re.compile(r"^https?://[^/]*\.blogspot\.[a-z.]+(?:[:/]|$)")
GROUPS = ("blogspot_google", "casino", "betting")


def label_group(target: str | None, url_clean: str) -> str:
    """피싱 행의 묶음 이름. 해당 없으면 'other'."""
    target = target.strip().lower() if isinstance(target, str) else ""
    if target == "google" and BLOGSPOT.match(url_clean or ""):
        return "blogspot_google"
    if target in CASINO:
        return "casino"
    if target in BETTING:
        return "betting"
    return "other"
