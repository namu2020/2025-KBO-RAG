import re, time, random, csv
from io import StringIO
from typing import Dict, List
import requests
from bs4 import BeautifulSoup
import pandas as pd

# ====== 설정 ======
HDRS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8",
    "Referer": "https://www.koreabaseball.com/",
}

# 네가 준 HTML 그대로
TEAM_SELECT_NAME = "ctl00$ctl00$ctl00$cphContents$cphContents$cphContents$ddlTeam$ddlTeam"
TEAM_OPTIONS = [  # (보여지는 이름, option value)
    ("LG", "LG"),
    ("한화", "HH"),
    ("SSG", "SK"),
    ("삼성", "SS"),
    ("KT", "KT"),
    ("NC", "NC"),
    ("롯데", "LT"),
    ("KIA", "HT"),
    ("두산", "OB"),
    ("키움", "WO"),
]

URLS = {
    "hitter": "https://www.koreabaseball.com/Record/Player/HitterBasic/Basic1.aspx",
    "pitcher": "https://www.koreabaseball.com/Record/Player/PitcherBasic/Basic1.aspx",
}

# ====== 유틸 ======
def wait():
    time.sleep(0.6 + random.random() * 0.7)

def make_soup(text: str) -> BeautifulSoup:
    for parser in ("lxml", "html5lib", "html.parser"):
        try:
            return BeautifulSoup(text, parser)
        except Exception:
            continue
    raise RuntimeError("No HTML parser available (install lxml/html5lib)")

def GET(session: requests.Session, url: str) -> BeautifulSoup:
    r = session.get(url, headers=HDRS, timeout=30)
    r.raise_for_status()
    return make_soup(r.text)

def POST(session: requests.Session, url: str, data: Dict[str, str]) -> BeautifulSoup:
    r = session.post(url, headers=HDRS, data=data, timeout=30)
    r.raise_for_status()
    return make_soup(r.text)

def collect_hidden_fields(soup: BeautifulSoup) -> Dict[str, str]:
    """
    페이지 내 모든 hidden input을 사전에 넣어 반환.
    WebForms는 __VIEWSTATE/VALIDATION 외에도 부가 hidden이 있을 수 있어 통째로 보냄.
    """
    payload = {}
    for inp in soup.select("input[type=hidden]"):
        name = inp.get("name")
        if not name:
            continue
        payload[name] = inp.get("value", "")
    # 기본 이벤트 필드 없으면 채움
    payload.setdefault("__EVENTTARGET", "")
    payload.setdefault("__EVENTARGUMENT", "")
    return payload

def read_table_df_from_page(soup: BeautifulSoup) -> pd.DataFrame:
    # 가장 큰 데이터 테이블을 우선 선택
    table = soup.select_one("table.tData01.tt") or soup.find("table")
    if not table:
        return pd.DataFrame()
    dfs = pd.read_html(StringIO(str(table)), flavor="bs4")
    if not dfs:
        return pd.DataFrame()
    df = dfs[0].copy()
    # 첫 행에 헤더가 들어왔으면 승격
    if len(df) > 0:
        first = "".join(df.iloc[0].astype(str).tolist())
        if any(k in first for k in ["선수명", "팀명", "AVG", "ERA"]):
            df.columns = df.iloc[0]
            df = df.iloc[1:].reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df

# ====== 새 페이지네이션 유틸 ======
def pager_actions(soup: BeautifulSoup):
    """
    div.paging 영역의 __doPostBack target/argument를 (label, target, arg, is_current) 리스트로 반환.
    숫자 버튼은 arg가 ''(빈 문자열)인 경우가 있음(예: ucPager$btnNo2).
    """
    out = []
    for a in soup.select("div.paging a[href*='__doPostBack']"):
        href = a.get("href", "")
        m = re.search(r"__doPostBack\('([^']+)'\s*,\s*'([^']*)'\)", href)
        if not m:
            continue
        target, arg = m.group(1), m.group(2)
        label = (a.get_text(strip=True) or a.get("title", "")).strip()  # '1','2','다음','처음으로' 등
        is_current = "on" in (a.get("class") or [])
        out.append((label, target, arg, is_current))
    return out

def pick_next_page(pager_list):
    """
    현재 페이지(class='on') 다음의 '숫자' 버튼을 우선 선택.
    없으면 '다음/Next' 버튼. 더 이상 없으면 None.
    """
    curr_idx = None
    indices_nums = []
    for i, (label, target, arg, is_curr) in enumerate(pager_list):
        if is_curr:
            curr_idx = i
        if label.isdigit():
            indices_nums.append(i)
    # 현재 다음 숫자
    if curr_idx is not None:
        for i in indices_nums:
            if i > curr_idx:
                _, tgt, arg, _ = pager_list[i]
                return tgt, arg
    # '다음' 계열
    for label, tgt, arg, _ in pager_list:
        if label in ("다음", "Next", ">"):
            return tgt, arg
    return None

# ====== 크롤 핵심 ======
def scrape_mode(mode: str):
    assert mode in URLS, "mode must be 'hitter' or 'pitcher'"
    base = URLS[mode]
    s = requests.Session()
    all_rows: List[Dict] = []

    for team_text, team_val in TEAM_OPTIONS:
        print(f"[{mode}] 팀 선택 → {team_text} ({team_val})")
        # 1) 첫 로드
        sp = GET(s, base)

        # 2) hidden 필드 수집 + 드롭다운 선택 포스트백
        payload = collect_hidden_fields(sp)
        payload.update({
            TEAM_SELECT_NAME: team_val,         # 선택값
            "__EVENTTARGET": TEAM_SELECT_NAME,  # 드롭다운이 이벤트 소스
            "__EVENTARGUMENT": "",
        })
        sp = POST(s, base, payload)
        wait()

        # 3) 페이지네이션 루프
        team_rows: List[Dict] = []
        visited = set()
        while True:
            df = read_table_df_from_page(sp)
            if not df.empty and ("팀명" in df.columns and "선수명" in df.columns):
                df["team_filter"] = team_text
                team_rows.extend(df.to_dict(orient="records"))
            else:
                print(f"  [WARN] 표 인식 실패 — {team_text}")

            # 페이저 분석 → 다음 페이지 결정
            actions = pager_actions(sp)
            # 방문한 target/arg는 제외
            actions = [(lbl, tgt, arg, cur) for (lbl, tgt, arg, cur) in actions if (tgt, arg) not in visited]
            nxt = pick_next_page(actions)
            if not nxt:
                break

            target, arg = nxt
            visited.add((target, arg))

            p2 = collect_hidden_fields(sp)   # 매 페이지 최신 hidden 재수집
            p2.update({
                "__EVENTTARGET": target,
                "__EVENTARGUMENT": arg,      # 숫자 버튼이면 대부분 '' 여도 OK
                TEAM_SELECT_NAME: team_val,  # 팀 선택 유지!
            })
            sp = POST(s, base, p2)
            wait()

        # 4) 팀별 저장
        if team_rows:
            save_csv(team_rows, f"{mode}_basic_{team_text}.csv")
            all_rows.extend(team_rows)
        else:
            print(f"  !! No rows for {team_text}")

    # 5) 통합 저장
    if all_rows:
        save_csv(all_rows, f"{mode}_basic_ALL.csv")
    else:
        print(f"!! No rows for {mode} (ALL)")

def save_csv(rows: List[Dict], path: str):
    cols = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    print("Saved:", path, len(rows))

if __name__ == "__main__":
    scrape_mode("hitter")
    scrape_mode("pitcher")
