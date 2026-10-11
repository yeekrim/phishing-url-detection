# v2 2단계: 피싱·정상 URL 에 접속해 피처를 계산하는 격리 환경
FROM python:3.12-slim

# 루트가 아닌 사용자로 실행
RUN useradd --create-home --uid 10001 crawler
WORKDIR /work

COPY docker/requirements-v2.txt /tmp/requirements-v2.txt
RUN pip install --no-cache-dir -r /tmp/requirements-v2.txt && rm /tmp/requirements-v2.txt

USER crawler
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# 코드와 데이터는 실행할 때 볼륨으로 붙인다 (run_build_features.sh)
ENTRYPOINT ["python", "src/v2/build_features.py"]
