#!/usr/bin/env bash
# 컨테이너 안에서 피처를 계산한다. 레포 루트에서 실행:
#   bash docker/run_build_features.sh --tag pilot --phishing-sample 2000 --legit-sample 2000
#
# 격리 설정
#   - 코드(src/)는 읽기 전용으로 붙이고, 쓰기는 data/v2/interim·processed 에만 허용
#   - 컨테이너 파일시스템은 읽기 전용, /tmp 만 메모리에 임시로 허용
#   - 모든 리눅스 권한 제거, 권한 상승 금지, CPU·메모리·프로세스 수 제한
#   - 루트가 아닌 사용자(uid 10001)로 실행
# 사설망 주소 접속 차단은 코드(src/v2/fetch.py)에서 한다.
set -euo pipefail
IMAGE=phishing-v2-features

docker build -q -t "$IMAGE" -f docker/Dockerfile . >/dev/null
mkdir -p data/v2/interim data/v2/processed

docker run --rm \
  --read-only --tmpfs /tmp:rw,size=256m \
  --cap-drop ALL --security-opt no-new-privileges \
  --memory 2g --cpus 2 --pids-limit 256 \
  --user 10001:10001 \
  -v "$PWD/src:/work/src:ro" \
  -v "$PWD/data/v2/interim:/work/data/v2/interim" \
  -v "$PWD/data/v2/processed:/work/data/v2/processed" \
  "$IMAGE" "$@"
