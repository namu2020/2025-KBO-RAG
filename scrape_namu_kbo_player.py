# -*- coding: utf-8 -*-
"""
[나무위키 인물 문서 수집기] 팀별 선수명으로 본문 텍스트 저장
- CSV(타자/투수)에서 팀 필터 후 선수명 수집
- 팀 '.../선수단' 페이지의 링크를 우선 사용 (동명이인 방지 목적)
- 그래도 동명이인/모호 페이지로 떨어지면: '야구/KBO/팀명' 기준으로 후보를 순회해 재해결
- 본문 텍스트를 <팀>/<선수명>.txt 로 저장

필요: requests, beautifulsoup4
pip install requests beautifulsoup4
"""

# =========================
# Config (여기만 바꾸면 됨)
# =========================
team = "KT"  # 'LG', '두산', '한화', 'KIA', '삼성', 'SSG', '키움', 'NC', '롯데', 'KT' 등
csv_files = [
    "hitter_basic_All.csv",
    "pitcher_basic_All.csv",
]
out_root = "./namu_people_txt"  # 저장 루트 폴더
use_team_page_mapping = True    # 팀 선수단 페이지 앵커 맵 우선 사용
delay_base_sec = 1.0            # 요청 간 평균 지연(초)

# =========================
# 본 코드
# =========================
import os, re, csv, time, random
from typing import List, Dict, Tuple, Optional
from urllib.parse import urljoin, quote
import requests
from bs4 import BeautifulSoup
import difflib

BASE = "https://namu.wiki"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

TEAM_FULL_NAME = {
    "LG": "LG 트윈스",
    "두산": "두산 베어스",
    "한화": "한화 이글스",
    "KIA": "KIA 타이거즈",
    "삼성": "삼성 라이온즈",
    "SSG": "SSG 랜더스",
    "키움": "키움 히어로즈",
    "NC": "NC 다이노스",
    "롯데": "롯데 자이언츠",
    "KT": "KT 위즈",
}

session = requests.Session()
session.headers.update({"User-Agent": UA, "Accept-Language": "ko,en;q=0.8"})

# -------------------- 공통 유틸 --------------------
def fetch_html(url: str, max_retry: int = 3, sleep=(0.7, 1.5)) -> str:
    last = None
    for _ in range(max_retry):
        resp = session.get(url, timeout=25)
        last = resp.status_code
        if resp.status_code == 200:
            return resp.text
        time.sleep(random.uniform(*sleep))
    raise RuntimeError(f"GET 실패: {url} (status={last})")

def clean_text(s: str) -> str:
    s = re.sub(r'\u200b|\xa0|\r', ' ', s)
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()

def pick_main_text(soup: BeautifulSoup) -> str:
    candidates = [
        ("article", {}),
        ("div", {"class": "wiki-article"}),
        ("div", {"id": "content"}),
        ("main", {}),
    ]
    for name, attrs in candidates:
        node = soup.find(name, attrs)
        if not node:
            continue
        for bad in node.select(".toc, .footnotes, nav, header, .ad, .advertisement"):
            bad.decompose()
        txt = node.get_text("\n", strip=True)
        if len(txt) > 200:
            return clean_text(txt)
    return clean_text(soup.get_text("\n", strip=True))

def extract_page_name(soup: BeautifulSoup, default_name="문서") -> str:
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return og["content"].strip()
    title = soup.find("title")
    if title and title.get_text(strip=True):
        t = re.sub(r"\s*-\s*나무위키$", "", title.get_text(strip=True))
        return t or default_name
    return default_name

def safe_filename(name: str) -> str:
    for a,b in [("/", "／"), ("\\","＼"), (":","："), ("*","＊"),
                ("?","？"), ('"',"＂"), ("<","＜"), (">","＞"), ("|","｜")]:
        name = name.replace(a,b)
    return (name.strip() or "문서")

# ---------- [헬퍼] 이름-제목 크로스체크 ----------
def _norm(s: str) -> str:
    """공백 제거 + 소문자화 (한글엔 영향 없고 영문/공백 차이 흡수)"""
    return re.sub(r"\s+", "", (s or "")).lower()

def _title_has_name(name: str, page_title: str) -> bool:
    """
    페이지명에 선수명이 '포함'되는지 판단.
    예: name='신민재' → page_title='신민재(야구선수)' True
        name='오스틴' → page_title='오스틴 딘' True
    """
    return _norm(name) in _norm(page_title)

# -------------------- 팀 페이지 링크 수집 --------------------
def looks_like_person_loose(text: str) -> bool:
    if not text:
        return False
    # 한글/알파벳이 하나라도 있으면 패스 (괄호/숫자 허용)
    return bool(re.search(r"[가-힣A-Za-z]", text))

def _strip_paren(s: str) -> str:
    # 괄호 안 내용 제거 (여러 개도 제거)
    return re.sub(r"\s*\(.*?\)\s*", "", s).strip()

def _keys_from_anchor(a) -> tuple[list, str]:
    """앵커로부터 매핑 키 후보들과 절대 URL을 뽑는다"""
    href = a.get("href") or ""
    if not href:
        return [], ""
    abs_url = urljoin(BASE, href)

    t_text = (a.get_text(strip=True) or "").strip()
    t_title = (a.get("title") or "").strip()

    cand = set()
    for s in (t_text, t_title):
        if s:
            cand.add(s)
            cand.add(_strip_paren(s))
    # 외인 성명 공백 분리(오스틴 딘 → 오스틴)
    for s in list(cand):
        if " " in s:
            cand.add(s.split()[0])

    # 노이즈 제거: 한글/알파벳 없는 키 제외
    keys = [k for k in cand if re.search(r"[가-힣A-Za-z]", k)]
    return keys, abs_url

def build_name_to_url_map(team_short: str) -> Dict[str, str]:
    """팀 선수단 페이지에서 '표시이름/변형 -> href' 매핑(절대/상대 URL 모두)"""
    full = TEAM_FULL_NAME.get(team_short, team_short)
    team_page = "https://namu.wiki/w/kt%20wiz/%EC%84%A0%EC%88%98%EB%8B%A8"
    html = fetch_html(team_page)
    soup = BeautifulSoup(html, "html.parser")
    mapping: Dict[str, str] = {}

    article = soup.find("article") or soup
    for a in article.select('a[href^="/w/"], a[href^="https://namu.wiki/w/"]'):
        keys, abs_url = _keys_from_anchor(a)
        if not abs_url:
            continue
        # 불필요한 시스템/토론 링크 제외
        if any(seg in abs_url for seg in ("/discuss", "/Recent", "/ACL", "/Random", "/history")):
            continue
        for k in keys:
            if looks_like_person_loose(k):
                mapping.setdefault(k, abs_url)
    return mapping

def _fuzzy_pick_url(name: str, name2url_map: Dict[str, str]) -> tuple[str|None, str]:
    """정확 일치가 없으면 퍼지 매칭으로 URL 선택"""
    if not name2url_map:
        return None, ""
    keys = list(name2url_map.keys())
    norm = _strip_paren(name)

    # 1) 완전 일치/괄호 제거 일치
    for k in (name, norm):
        if k in name2url_map:
            return name2url_map[k], "team-map"

    # 2) startswith / contains 우선
    starts = [k for k in keys if k.startswith(name)]
    contains = [k for k in keys if name in k]
    for cand in (starts + contains):
        return name2url_map[cand], "team-map≈fuzzy"

    # 3) difflib 근사
    near = difflib.get_close_matches(name, keys, n=1, cutoff=0.8)
    if near:
        return name2url_map[near[0]], "team-map≈fuzzy"

    return None, ""

# -------------------- CSV 로딩 --------------------
def load_names_from_csv_filtered(csv_paths: List[str],
                                 team_short: str,
                                 name_col: str = "선수명",
                                 team_col: str = "팀명") -> List[str]:
    names: List[str] = []
    for path in csv_paths:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if name_col not in reader.fieldnames or team_col not in reader.fieldnames:
                raise ValueError(f"{path}: 필요한 컬럼 없음. 헤더={reader.fieldnames}, 필요='{name_col}','{team_col}'")
            for row in reader:
                t = (row.get(team_col) or "").strip()
                if t and team_short in t:
                    nm = (row.get(name_col) or "").strip()
                    if nm:
                        names.append(nm)
    seen, uniq = set(), []
    for n in names:
        if n not in seen:
            seen.add(n); uniq.append(n)
    return uniq

# -------------------- 동명이인/검증 로직 --------------------
def is_disambiguation_page(soup: BeautifulSoup) -> bool:
    head_txt = (soup.find("article") or soup).get_text(" ", strip=True)[:600]
    if re.search(r"동음이의어|동명이인", head_txt):
        return True
    if re.search(r"다음(과|의)\s+.+\s+가리킬", head_txt):
        return True
    # 상단 다량 목록 휴리스틱
    if len((soup.select("article ul li") or [])[:12]) >= 6 and "분류" not in head_txt:
        return True
    return False

def _article_text(soup: BeautifulSoup) -> str:
    return (soup.find("article") or soup).get_text(" ", strip=True)

def text_mentions_team(soup: BeautifulSoup, team_full: str) -> bool:
    """본문/인포박스에 팀명이 등장하는지 간단 검증"""
    article = soup.find("article") or soup
    txt = article.get_text(" ", strip=True)
    return team_full in txt

def resolve_from_disambig(name: str, soup: BeautifulSoup, team_full: str) -> Optional[Tuple[str, BeautifulSoup]]:
    """동명이인 페이지에서 팀/야구 키워드 기반으로 올바른 문서 재해결"""
    article = soup.find("article") or soup

    def score_link(a) -> int:
        t = a.get_text(strip=True)
        h = a.get("href") or ""
        s = 0
        if team_full in t or team_full in h: s += 5
        if re.search(r"(야구|야구\s*선수|KBO)", t): s += 3
        if re.search(r"%EC%95%BC%EA%B5%AC|KBO", h): s += 2
        if "(" in t and ")" in t: s += 1
        return s

    cand = []
    for a in article.select('a[href^="/w/"], a[href^="https://namu.wiki/w/"]'):
        href = a.get("href", "")
        if not href or any(seg in href for seg in ("/discuss", "/Recent", "/ACL", "/Random", "/history")):
            continue
        sc = score_link(a)
        if sc > 0:
            cand.append((sc, urljoin(BASE, href)))

    # 중복 제거 & 고득점 우선
    seen, uniq = set(), []
    for sc, u in sorted(cand, key=lambda x: -x[0]):
        if u not in seen:
            seen.add(u); uniq.append(u)

    # 후보 순회: 팀명 포함 + 제목에 '선수명' 포함되는 페이지만 채택
    for url in uniq[:10]:
        try:
            html2 = fetch_html(url)
            s2 = BeautifulSoup(html2, "html.parser")
            page_title = extract_page_name(s2, default_name=name)
            txt = (s2.find("article") or s2).get_text(" ", strip=True)
            if (team_full in txt) and _title_has_name(name, page_title):
                return url, s2
        except Exception:
            continue

    # 표제 규칙 추정(보조 루트)
    for suffix in ["(야구 선수)", "(야구선수)", "(야구)", f"({team_full})"]:
        try:
            url = urljoin(BASE, "/w/" + quote(name + suffix, safe=""))
            html2 = fetch_html(url)
            s2 = BeautifulSoup(html2, "html.parser")
            page_title = extract_page_name(s2, default_name=name)
            txt = (s2.find("article") or s2).get_text(" ", strip=True)
            if (team_full in txt) and _title_has_name(name, page_title):
                return url, s2
        except Exception:
            pass
    return None

# -------------------- 핵심 파서 --------------------
def parse_person_by_name(name: str, name2url_map: Dict[str, str] | None, team_short: str):
    """
    1) team-map(정확/퍼지)로 URL 확보 시, 먼저 해당 페이지 제목에 '선수명' 포함되면 즉시 확정(해결 절대 금지)
    2) team-map인데 제목에 선수명 미포함 → 그때만 재시도 (direct/resolve)
    3) 어떤 경로든 '최종 저장 직전'에도 제목-이름 크로스체크. 불일치면 실패 처리
    """
    team_full = TEAM_FULL_NAME.get(team_short, team_short)

    # --- URL 선택: team-map(퍼지 포함) → 없으면 direct ---
    url, source = None, ""
    if name2url_map:
        # 정확/퍼지 매칭 시도
        url, source = _fuzzy_pick_url(name, name2url_map)
    if not url:
        url = urljoin(BASE, "/w/" + quote(name, safe=""))
        source = "direct"

    # --- 최초 페이지 로드 ---
    html = fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    page_name = extract_page_name(soup, default_name=name)

    # --- [중요] team-map으로 들어왔고, 제목에 선수명이 포함되면 그대로 확정 (resolve 금지) ---
    if source.startswith("team-map") and _title_has_name(name, page_name):
        body_text = pick_main_text(soup)
        # 최종 안전벨트: 그래도 팀 문서 같은 엉뚱한 경우 막기 위해 한 번 더 확인
        if not _title_has_name(name, page_name):
            raise RuntimeError(f"제목-이름 불일치(팀맵 확정 단계): {page_name} vs {name}")
        return page_name, body_text, source  # ← 여기서 끝!

    # --- team-map인데 제목이 선수명 포함 안 되면: 보조 루트 시도 ---
    if source.startswith("team-map") and not _title_has_name(name, page_name):
        # 1) 동명이인 페이지면 재해결 시도
        if is_disambiguation_page(soup):
            resolved = resolve_from_disambig(name, soup, team_full)
            if resolved:
                url, soup = resolved
                page_name = extract_page_name(soup, default_name=name)
                source = f"{source}->resolved"
        # 2) 그래도 불안하면 /w/<이름> 직접 진입해서 다시 확인
        if not _title_has_name(name, page_name):
            try:
                url2 = urljoin(BASE, "/w/" + quote(name, safe=""))
                html2 = fetch_html(url2)
                s2 = BeautifulSoup(html2, "html.parser")
                page_name2 = extract_page_name(s2, default_name=name)
                if _title_has_name(name, page_name2):
                    soup, page_name = s2, page_name2
                    source = f"{source}->direct"
            except Exception:
                pass

    # --- direct로 출발한 경우: 필요 시 동명이인 재해결 ---
    if source == "direct" and is_disambiguation_page(soup):
        resolved = resolve_from_disambig(name, soup, team_full)
        if resolved:
            url, soup = resolved
            page_name = extract_page_name(soup, default_name=name)
            source = f"{source}->resolved"

    # --- 최종 추출 & 저장 전 최종 크로스체크 ---
    body_text = pick_main_text(soup)
    if not _title_has_name(name, page_name):
        # 엉뚱한 문서 저장 차단
        raise RuntimeError(f"제목-이름 불일치(최종): {page_name} vs {name}")

    return page_name, body_text, source

# -------------------- 메인 --------------------
def main():
    out_dir = os.path.join(out_root, team)
    os.makedirs(out_dir, exist_ok=True)

    names = load_names_from_csv_filtered(csv_files, team_short=team)
    print(f"팀 '{team}' 대상 선수명 {len(names)}명")

    name2url_map = None
    if use_team_page_mapping:
        try:
            name2url_map = build_name_to_url_map(team)
            print(f"- 팀 페이지 앵커 매핑 {len(name2url_map)}건 확보")
        except Exception as e:
            print(f"- 팀 페이지 매핑 실패: {e}")

    failed = []
    for i, name in enumerate(names, 1):
        try:
            time.sleep(delay_base_sec * random.uniform(0.6, 1.4))
            page_name, body_text, how = parse_person_by_name(name, name2url_map, team)
            out_path = os.path.join(out_dir, safe_filename(name) + ".txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(body_text)
            print(f"[{i}/{len(names)}] 저장: {out_path} (문자수 {len(body_text):,}) ← 페이지명: {page_name} [{how}]")
        except Exception as e:
            print(f"[{i}/{len(names)}] 실패: {name} → {e}")
            failed.append(name)

    # 무결성 체크
    missing = [n for n in names if not os.path.exists(os.path.join(out_dir, safe_filename(n) + ".txt"))]
    print(f"\n✅ 수집 성공: {len(names) - len(missing)} / {len(names)}")
    if failed or missing:
        union, seen = [], set()
        for n in (failed + missing):
            if n not in seen:
                seen.add(n); union.append(n)
        print("❌ 미수집/실패 목록:", union)
    else:
        print("🎉 모든 대상이 정상 저장되었습니다.")

if __name__ == "__main__":
    main()
