# -*- coding: utf-8 -*-
"""
[후처리] 나무위키 파싱 텍스트 전처리:
- 모든 txt에서 "다른 KBO 리그 팀 명단 보기" 최초 등장 이전 텍스트 삭제(머리 자르기)
- 모든 txt에서 [편집] 토큰(전각/공백 변형 포함) 일괄 제거
- 모든 txt에서 "이 저작물은 ... CC BY-NC-SA 2.0 KR" 등장 시, 그 문구 '포함'하여 이후 전부 삭제(꼬리 자르기)
- 백업/드라이런/CSV 리포트로 무결성 확인
"""

# =========================
# Config
# =========================
ROOT_DIR = "./namu_people_txt"                    # txt들이 들어있는 루트 폴더
KEEP_MARKER = True                                 # True면 머리 마커 포함해서 남김, False면 마커까지 삭제
RECURSIVE = True                                   # 하위 폴더 재귀 처리
DRY_RUN = False                                    # True면 쓰기 안 하고 리포트만 생성
MAKE_BACKUP = True                                 # True면 원본 백업 보관

# 머리(앞부분 삭제) 기준 마커 — 여러 변형을 넣을 수 있음
MARKERS = [
    "다른 KBO 리그 팀 명단 보기",
    "다른 KBO 리그 구단 명단 보기",
    "다른 KBO리그 팀 명단 보기",
    "타 KBO 리그 구단 명단 보기",
    "다른 KBO 리그 팀별 명단 둘러보기",
]

# 일괄 제거할 토큰(정규식 리스트)
# - 일반 대괄호 [] / 전각 대괄호 ［］ 모두 지원
# - 안쪽 공백 허용: [ 편집 ], ［  편집  ］ 등
STRIP_TOKENS_REGEX = [
    r"[［\[]\s*편집\s*[］\]]",     # [편집], ［편집］, 공백 변형 포함
    "다른 KBO 리그 팀 명단 보기",
    "다른 KBO 리그 구단 명단 보기",
    "다른 KBO리그 팀 명단 보기",
    "타 KBO 리그 구단 명단 보기",
    "다른 KBO 리그 팀별 명단 둘러보기",
    "[ 펼치기 · 접기 ]",
    r"[［\[]\s*[0-9０-９]+(?:\s*[-–~·,]\s*[0-9０-９]+)*\s*[］\]]"
]

# 꼬리(이후 전부 삭제) 기준 패턴(정규식) — "문구를 포함해서" 잘라냄
# 줄바꿈/공백/전각 등 변형을 허용하도록 넉넉하게 매칭
SUFFIX_PATTERNS = [
    r"이\s*저작물은[\s\S]{0,200}?CC\s*BY\-NC\-SA\s*2\.0\s*KR",  # 핵심 라이선스 문구
]

REPORT_BASENAME = "_strip_report"

# =========================
# Code
# =========================
import os, csv, re, hashlib, datetime, shutil

def norm_space(s: str) -> str:
    s = s.replace("\u00A0", " ")  # NBSP -> space
    s = s.replace("\u200b", "")   # zero-width space 제거
    s = re.sub(r"[ \t]+", " ", s)
    return s

def find_first_marker(text: str, markers: list[str]) -> tuple[str|None, int]:
    base = norm_space(text)
    earliest_idx = -1
    earliest_marker = None
    for m in markers:
        m_norm = norm_space(m)
        i = base.find(m_norm)
        if i != -1 and (earliest_idx == -1 or i < earliest_idx):
            earliest_idx = i
            earliest_marker = m_norm
    return earliest_marker, earliest_idx

def find_first_suffix_span(text: str, patterns: list[str]) -> tuple[int, int, str]|tuple[None, None, None]:
    """
    패턴 중 가장 먼저 매칭되는 구간을 반환: (start, end, pattern)
    - 패턴은 '문구 포함 삭제'이므로 end는 match.end()
    - 개행 포함 매칭을 위해 DOTALL, 영어 라이선스 대소문자 허용을 위해 IGNORECASE
    """
    earliest = None  # (start, end, pat)
    for pat in patterns:
        m = re.search(pat, text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            span = (m.start(), m.end(), pat)
            if (earliest is None) or (span[0] < earliest[0]):
                earliest = span
    return earliest if earliest else (None, None, None)

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def remove_tokens(text: str, regex_list: list[str]) -> tuple[str, int]:
    """정규식 토큰들을 전부 제거하고 (새텍스트, 제거개수) 반환"""
    removed = 0
    new_text = text
    for pat in regex_list:
        cnt = len(re.findall(pat, new_text))
        if cnt:
            new_text = re.sub(pat, "", new_text)
            removed += cnt
    return new_text, removed

def process_file(path: str, marker_list: list[str], keep_marker: bool,
                 backup_root: str|None, token_regex: list[str], suffix_patterns: list[str]):
    """
    파일 하나 처리:
    1) (있으면) 머리 마커 이전 텍스트 삭제
    2) [편집] 등 토큰 일괄 제거
    3) (있으면) 라이선스 꼬리 문구 '포함'하여 이후 전부 삭제
    4) 백업/쓰기/검증
    5) 리포트 dict 반환
    """
    rel = os.path.relpath(path, ROOT_DIR)
    with open(path, "rb") as f:
        raw_before = f.read()
    try:
        text = raw_before.decode("utf-8")
    except UnicodeDecodeError:
        return {
            "file": rel, "status": "encoding_error", "reason": "utf-8 decode fail",
            "len_before": len(raw_before), "len_after": None,
            "removed_total": None, "removed_prefix": None, "removed_tokens": None, "removed_suffix": None,
            "marker_found": False, "marker_index": None, "kept_marker": keep_marker,
            "suffix_found": False, "suffix_index": None, "suffix_pattern": "",
            "tokens_patterns": "|".join(token_regex),
            "sha_before": sha256_bytes(raw_before), "sha_after": None
        }

    # 1) 머리 마커 컷
    marker, idx = find_first_marker(text, marker_list)
    if marker is None:
        cur_text = text
        removed_prefix = 0
        marker_found = False
        marker_index = -1
    else:
        cut = idx if keep_marker else (idx + len(marker))
        cur_text = text[cut:]
        removed_prefix = len(text) - len(cur_text)
        marker_found = True
        marker_index = idx

    # 2) 토큰 제거
    cur_text, removed_tokens = remove_tokens(cur_text, token_regex)

    # 3) 꼬리(라이선스) 컷: 문구 '포함'해서 이후 전부 삭제
    s_start, s_end, s_pat = find_first_suffix_span(cur_text, suffix_patterns)
    if s_start is not None:
        new_text = cur_text[:s_start]  # 포함 삭제
        removed_suffix = len(cur_text) - len(new_text)
        suffix_found = True
        suffix_index = s_start
        cur_text = new_text
    else:
        removed_suffix = 0
        suffix_found = False
        suffix_index = -1
        s_pat = ""

    # 변경 여부
    changed = (cur_text != text)

    # 백업/쓰기
    if changed and (not DRY_RUN):
        if backup_root:
            backup_path = os.path.join(backup_root, rel)
            ensure_dir(os.path.dirname(backup_path))
            with open(backup_path, "wb") as bf:
                bf.write(raw_before)
        with open(path, "w", encoding="utf-8") as wf:
            wf.write(cur_text)

    # 검증
    if changed and (not DRY_RUN):
        with open(path, "r", encoding="utf-8") as rf:
            verify = rf.read()
        status = "ok" if verify == cur_text else "verify_mismatch"
    else:
        # 변경이 전혀 없으면 상태 표시
        if (marker is None and removed_tokens == 0 and removed_suffix == 0):
            status = "marker_not_found"
        else:
            status = "no_change" if (removed_prefix == 0 and removed_tokens == 0 and removed_suffix == 0) else "ok"

    return {
        "file": rel, "status": status, "reason": "",
        "len_before": len(text), "len_after": len(cur_text),
        "removed_total": len(text) - len(cur_text),
        "removed_prefix": removed_prefix,
        "removed_tokens": removed_tokens,
        "removed_suffix": removed_suffix,
        "marker_found": marker_found, "marker_index": marker_index,
        "kept_marker": keep_marker,
        "suffix_found": suffix_found, "suffix_index": suffix_index, "suffix_pattern": s_pat,
        "tokens_patterns": "|".join(token_regex),
        "sha_before": sha256_text(text), "sha_after": sha256_text(cur_text)
    }

def main():
    # 백업 폴더/리포트 파일 준비
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = None
    if MAKE_BACKUP and not DRY_RUN:
        backup_root = os.path.join(ROOT_DIR, f"_backup_strip_{ts}")
        ensure_dir(backup_root)

    report_path = os.path.join(ROOT_DIR, f"{REPORT_BASENAME}_{ts}.csv")

    # 타깃 파일 수집
    targets = []
    if RECURSIVE:
        for root, _, files in os.walk(ROOT_DIR):
            for fn in files:
                if fn.lower().endswith(".txt"):
                    targets.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(ROOT_DIR):
            if fn.lower().endswith(".txt"):
                targets.append(os.path.join(ROOT_DIR, fn))

    results = []
    total = len(targets)
    print(f"대상 파일: {total}개")

    for i, p in enumerate(targets, 1):
        r = process_file(p, MARKERS, KEEP_MARKER, backup_root, STRIP_TOKENS_REGEX, SUFFIX_PATTERNS)
        results.append(r)
        print(f"[{i}/{total}] {r['status']:>16}  "
              f"removed_prefix={r.get('removed_prefix')}, "
              f"removed_tokens={r.get('removed_tokens')}, "
              f"removed_suffix={r.get('removed_suffix')}  file={r['file']}")

    # 요약
    ok   = sum(1 for r in results if r["status"] == "ok")
    nf   = sum(1 for r in results if r["status"] == "marker_not_found")
    mis  = sum(1 for r in results if r["status"] == "verify_mismatch")
    enc  = sum(1 for r in results if r["status"] == "encoding_error")
    nochg= sum(1 for r in results if r["status"] == "no_change")
    tok_total = sum(r.get("removed_tokens") or 0 for r in results)
    pre_total = sum(r.get("removed_prefix") or 0 for r in results)
    suf_total = sum(r.get("removed_suffix") or 0 for r in results)

    print("\n=== Summary ===")
    print(f"OK                 : {ok}")
    print(f"Marker not found   : {nf}")
    print(f"Verify mismatch    : {mis}")
    print(f"Encoding error     : {enc}")
    print(f"No change          : {nochg}")
    print(f"Removed prefix sum : {pre_total:,} chars")
    print(f"Removed token cnt  : {tok_total:,} occurrences")
    print(f"Removed suffix sum : {suf_total:,} chars")
    print(f"Backup dir         : {backup_root or '(no backup)'}")
    print(f"Report CSV         : {report_path}")

    # CSV 리포트 저장
    fieldnames = [
        "file", "status", "reason",
        "len_before", "len_after",
        "removed_total", "removed_prefix", "removed_tokens", "removed_suffix",
        "marker_found", "marker_index", "kept_marker",
        "suffix_found", "suffix_index", "suffix_pattern",
        "tokens_patterns",
        "sha_before", "sha_after"
    ]
    with open(report_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

if __name__ == "__main__":
    main()
