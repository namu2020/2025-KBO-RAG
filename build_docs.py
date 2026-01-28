# build_docs.py
import os, glob, json, chardet, pandas as pd
from typing import List, Optional, Dict, Any
from hashlib import md5

OUT_JSONL = "./artifacts/raw_docs.jsonl"  # 결과물
OUT_NAMES = "./artifacts/player_names.json"

try:
    # langchain < 0.2
    from langchain.schema import Document
except Exception:
    # langchain >= 0.2
    from langchain_core.documents import Document


class SportsWikiHybridLoader:
    def __init__(
        self,
        stats_csv_dir: str,
        wiki_txt_dir: str,
        csv_glob: str = "**/*.csv",
        txt_glob: str = "**/*.txt",
        encoding_fallbacks: tuple = ("utf-8", "cp949", "euc-kr"),
        player_col_candidates: tuple = ("선수", "선수명", "이름", "선수이름", "Player", "Name"),
        year_col_candidates: tuple = ("연도", "시즌", "Year", "YEAR"),
        team_col_candidates: tuple = ("팀", "팀명", "구단", "Team"),
        position_col_candidates: tuple = ("포지션", "Position", "POS"),
    ):
        self.stats_csv_dir = stats_csv_dir
        self.wiki_txt_dir = wiki_txt_dir
        self.csv_glob = csv_glob
        self.txt_glob = txt_glob
        self.encoding_fallbacks = encoding_fallbacks
        self.player_col_candidates = player_col_candidates
        self.year_col_candidates = year_col_candidates
        self.team_col_candidates = team_col_candidates
        self.position_col_candidates = position_col_candidates

    def load(self) -> List[Document]:
        docs: List[Document] = []
        docs.extend(self._load_csv_docs())
        docs.extend(self._load_txt_docs())
        return docs

    # ---------- helpers ----------
    def _detect_encoding(self, path: str) -> str:
        with open(path, "rb") as f:
            raw = f.read(4096)
        guess = chardet.detect(raw or b"")
        enc = (guess.get("encoding") or "").lower()
        return enc or self.encoding_fallbacks[0]

    def _pick_col(self, columns: List[str], candidates: tuple) -> Optional[str]:
        lower = {c.lower(): c for c in columns}
        for cand in candidates:
            if cand in columns:
                return cand
            if cand.lower() in lower:
                return lower[cand.lower()]
        return None

    def _row_to_text(self, row: pd.Series, order_hint: Optional[List[str]] = None) -> str:
        keys = order_hint or list(row.index)
        parts = []
        for k in keys:
            if k in row and pd.notna(row[k]):
                parts.append(f"{k}: {row[k]}")
        return "\n".join(parts)

    def _apply_boost(self, page_text: str, player: Optional[str], team: Optional[str]) -> str:
        """문서 상단에 도메인 태그 + 선수/팀 부스팅을 넣고 e5/bge용 'passage:' 프리픽스를 적용."""
        boost = ["[KBO][야구][선수]"]
        if player:
            boost += [f"선수:{player}", f"{player}"]  # 이름 2회 노출
        if team:
            boost += [f"팀:{team}"]
        boost_header = " ".join(boost).strip()
        return f"passage: {boost_header}\n{page_text}"

    def _load_csv_docs(self) -> List[Document]:
        docs: List[Document] = []
        for path in glob.glob(os.path.join(self.stats_csv_dir, self.csv_glob), recursive=True):
            # 인코딩 탐지 + 폴백
            df = None
            tried = set()
            for enc in (self._detect_encoding(path), *self.encoding_fallbacks):
                if enc in tried:
                    continue
                tried.add(enc)
                try:
                    df = pd.read_csv(path, encoding=enc)
                    break
                except Exception:
                    continue
            if df is None or df.empty:
                continue

            # 주요 컬럼 자동 감지
            player_col = self._pick_col(df.columns.tolist(), self.player_col_candidates)
            year_col   = self._pick_col(df.columns.tolist(), self.year_col_candidates)
            team_col   = self._pick_col(df.columns.tolist(), self.team_col_candidates)
            pos_col    = self._pick_col(df.columns.tolist(), self.position_col_candidates)

            order_hint = [c for c in [player_col, year_col, team_col, pos_col] if c] + \
                         [c for c in df.columns if c not in {player_col, year_col, team_col, pos_col}]

            # 각 행 -> Document
            for _, r in df.iterrows():
                page_text = self._row_to_text(r, order_hint=order_hint)
                meta: Dict[str, Any] = {
                    "type": "stats",
                    "source": path,
                    "player": (str(r[player_col]).strip() if player_col and pd.notna(r.get(player_col)) else None),
                    "year":   (int(r[year_col]) if year_col and pd.notna(r.get(year_col)) else None),
                    "team":   (str(r[team_col]).strip() if team_col and pd.notna(r.get(team_col)) else None),
                    "position": (str(r[pos_col]).strip() if pos_col and pd.notna(r.get(pos_col)) else None),
                    "all_columns": list(df.columns),
                    "filename": os.path.basename(path),
                }
                # ✅ CSV에도 프리픽스 + 부스팅 적용 (inferred_player 사용 금지)
                page_text = self._apply_boost(page_text, meta.get("player"), meta.get("team"))
                docs.append(Document(page_content=page_text, metadata=meta))
        return docs

    def _load_txt_docs(self) -> List[Document]:
        docs: List[Document] = []
        for path in glob.glob(os.path.join(self.wiki_txt_dir, self.txt_glob), recursive=True):
            base = os.path.splitext(os.path.basename(path))[0]
            inferred_player = base.strip()

            text = None
            for enc in (self._detect_encoding(path), *self.encoding_fallbacks):
                try:
                    with open(path, "r", encoding=enc, errors="ignore") as f:
                        t = f.read()
                    t = t.replace("\x00", "")  # 널문자 제거
                    text = t
                    break
                except Exception:
                    continue
            if text is None:
                text = ""

            header = f"[나무위키 문서]\n선수: {inferred_player}\n파일: {os.path.basename(path)}\n\n"
            page_text = header + text
            meta = {
                "type": "wiki",
                "source": path,
                "player": inferred_player,
                "filename": os.path.basename(path),
                "chars": len(text),
            }
            # ✅ TXT에도 프리픽스 + 부스팅 적용
            page_text = self._apply_boost(page_text, meta.get("player"), None)
            docs.append(Document(page_content=page_text, metadata=meta))
        return docs


def dump_jsonl(docs: List[Document], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for d in docs:
            rec = {
                "page_content": d.page_content,
                "metadata": d.metadata,
                "content_md5": md5(d.page_content.encode("utf-8")).hexdigest(),  # 변경 추적용 해시
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"✅ wrote {out_path} ({len(docs)} docs)")


def dump_player_names(docs: List[Document], out: str = OUT_NAMES):
    names = sorted({(d.metadata or {}).get("player") for d in docs if (d.metadata or {}).get("player")})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(names, f, ensure_ascii=False, indent=2)
    print(f"✅ wrote {out} ({len(names)} names)")


if __name__ == "__main__":
    loader = SportsWikiHybridLoader(
        stats_csv_dir="./2025csv",
        wiki_txt_dir="./namu_people_txt",
    )
    docs = loader.load()
    n_txt = sum(1 for d in docs if d.metadata.get("type") == "wiki")
    print(f"Loaded docs: total={len(docs)} (wiki={n_txt})")
    dump_jsonl(docs, OUT_JSONL)
    dump_player_names(docs, OUT_NAMES)
