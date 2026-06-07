#!/usr/bin/env bash
# ============================================================
#  SciFind — Test semua endpoint
#  Jalankan saat server lokal aktif: python api/index.py
#  atau: flask run --app api/index.py
# ============================================================

BASE="http://localhost:5000"
PASS=0; FAIL=0

check() {
  local label="$1"; local url="$2"; local data="$3"
  local method="${4:-GET}"
  if [ "$method" = "POST" ]; then
    resp=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
      -H "Content-Type: application/json" -d "$data" "$url")
  else
    resp=$(curl -s -o /dev/null -w "%{http_code}" "$url")
  fi
  if [ "$resp" = "200" ]; then
    echo "  PASS  [$resp] $label"
    ((PASS++))
  else
    echo "  FAIL  [$resp] $label"
    ((FAIL++))
  fi
}

echo ""
echo "========================================"
echo "  SciFind Endpoint Test"
echo "========================================"

check "GET  /api/health"          "$BASE/api/health"
check "GET  /api/search (TF-IDF)" "$BASE/api/search?query=space&method=tfidf"
check "GET  /api/search (hybrid)" "$BASE/api/search?query=alien+invasion"
check "POST /api/search"          "$BASE/api/search" '{"query":"robot","method":"hybrid"}' POST
check "GET  /api/semantic-search" "$BASE/api/semantic-search?query=time+travel"
check "POST /api/semantic-search" "$BASE/api/semantic-search" '{"query":"mars colony","top_k":5}' POST
check "GET  /api/recommend"       "$BASE/api/recommend?title=Dune&top_n=4"
check "POST /api/recommend"       "$BASE/api/recommend" '{"title":"Dune","top_n":4}' POST

echo ""
echo "========================================"
printf "  Result: %d passed, %d failed\n" $PASS $FAIL
echo "========================================"
echo ""
