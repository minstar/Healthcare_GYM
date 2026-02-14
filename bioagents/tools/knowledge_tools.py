"""Unified Knowledge Search & Browse Tools for BIOAgents.

Consolidates all search and browse capabilities into a single tool set that
any domain can use. Instead of each domain maintaining separate search tools,
all knowledge retrieval goes through this unified interface.

Knowledge Sources:
1. PubMed-style literature (simulated from db.json articles)
2. Medical Wiki / Encyclopedia (simulated from db.json wiki_entries)
3. Evidence Retrieval (MedCPT-style from db.json evidence_passages)
4. Clinical Guidelines Search
5. Wikipedia Dump (real FTS5/FAISS from wiki2018 dump)

Architecture:
    Domain Toolkit (e.g., ClinicalTools)
        └── includes KnowledgeTools as a mixin/composed tool set
            ├── search(queries)        → unified search across all sources
            ├── browse(url_or_id)      → browse any knowledge source
            ├── search_pubmed(query)   → PubMed-specific search
            ├── search_wiki(query)     → Wikipedia/encyclopedia search
            ├── search_evidence(query) → Evidence passage retrieval
            ├── search_guidelines(condition) → Clinical guidelines
            └── browse_article(id)     → Article/entry browsing
"""

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from bioagents.environment.toolkit import ToolKitBase, ToolType, is_tool


# ============================================================
# Wiki Backend (FTS5 + optional FAISS)
# ============================================================


class MedicalKnowledgeBackend:
    """Backend for searching indexed medical knowledge (MedCPT, PubMed, textbooks).

    Provides BM25 full-text search over:
    - MedCPT evidence passages (581K from PubMed/PMC)
    - Biomedical instructions (122K QA pairs)
    - Generator retrieval passages (83K)
    - MedInstruct-52k (52K)

    Total: ~828K searchable passages in passages_fts.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path
        self._conn: Optional[Any] = None
        self._initialized = False
        self._lock = threading.Lock()

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return self._conn is not None

        with self._lock:
            if self._initialized:
                return self._conn is not None

            try:
                import sqlite3
                db_path = self._resolve_db_path()
                if db_path is None or not os.path.exists(db_path):
                    logger.debug("Medical knowledge FTS index not found")
                    self._initialized = True
                    return False

                self._conn = sqlite3.connect(db_path)
                self._conn.execute("PRAGMA journal_mode=WAL;")
                self._initialized = True
                logger.info(f"Medical knowledge FTS backend initialized: {db_path}")
                return True
            except Exception as e:
                logger.warning(f"Failed to init medical knowledge backend: {e}")
                self._initialized = True
                return False

    def _resolve_db_path(self) -> Optional[str]:
        if self._db_path and os.path.exists(self._db_path):
            return self._db_path

        # Search common locations
        candidates = [
            os.path.join(os.path.dirname(__file__), "../../databases/medical_knowledge_fts.sqlite"),
            os.environ.get("MEDICAL_FTS_DB", ""),
        ]
        for c in candidates:
            c = os.path.abspath(c) if c else ""
            if c and os.path.exists(c):
                return c
        return None

    def search_passages(self, query: str, topk: int = 8, source: str = "") -> List[Dict[str, str]]:
        """Search unified passages via BM25."""
        if not self._ensure_initialized():
            return []

        terms = [t.lower() for t in re.findall(r"[A-Za-z0-9]{2,}", query)]
        stop = {"what", "which", "who", "when", "where", "why", "how",
                "is", "are", "was", "were", "the", "a", "an", "and", "or",
                "of", "to", "in", "on", "for", "with", "by", "from"}
        filtered = [t for t in terms if t not in stop][:12]
        if not filtered:
            filtered = terms[:8]
        fts_q = " OR ".join(filtered)
        if not fts_q:
            return []

        try:
            if source:
                cur = self._conn.execute(
                    "SELECT doc_id, source, title, snippet(passages_fts, 3, '<b>', '</b>', '...', 24), category, dataset_name "
                    "FROM passages_fts WHERE passages_fts MATCH ? AND source = ? "
                    "ORDER BY bm25(passages_fts) LIMIT ?",
                    (fts_q, source, topk),
                )
            else:
                cur = self._conn.execute(
                    "SELECT doc_id, source, title, snippet(passages_fts, 3, '<b>', '</b>', '...', 24), category, dataset_name "
                    "FROM passages_fts WHERE passages_fts MATCH ? "
                    "ORDER BY bm25(passages_fts) LIMIT ?",
                    (fts_q, topk),
                )
            results = []
            for doc_id, src, title, snippet, category, dataset_name in cur.fetchall():
                results.append({
                    "doc_id": str(doc_id or "").strip(),
                    "source": str(src or "").strip(),
                    "title": str(title or "").strip(),
                    "snippet": str(snippet or "").strip(),
                    "category": str(category or "").strip(),
                    "dataset_name": str(dataset_name or "").strip(),
                })
            return results
        except Exception as e:
            logger.debug(f"Medical FTS search error: {e}")
            return []

    def search_evidence(self, query: str, topk: int = 5) -> List[Dict[str, str]]:
        """Search MedCPT evidence specifically."""
        if not self._ensure_initialized():
            return []

        terms = [t.lower() for t in re.findall(r"[A-Za-z0-9]{2,}", query)]
        stop = {"what", "which", "who", "when", "where", "why", "how",
                "is", "are", "was", "were", "the", "a", "an", "and", "or",
                "of", "to", "in", "on", "for", "with", "by", "from"}
        filtered = [t for t in terms if t not in stop][:12]
        if not filtered:
            filtered = terms[:8]
        fts_q = " OR ".join(filtered)
        if not fts_q:
            return []

        try:
            cur = self._conn.execute(
                "SELECT doc_id, question, snippet(evidence_fts, 2, '<b>', '</b>', '...', 24), dataset_name "
                "FROM evidence_fts WHERE evidence_fts MATCH ? "
                "ORDER BY bm25(evidence_fts) LIMIT ?",
                (fts_q, topk),
            )
            results = []
            for doc_id, question, snippet, dataset_name in cur.fetchall():
                results.append({
                    "doc_id": str(doc_id or ""),
                    "source": "medcpt_evidence",
                    "question": str(question or "")[:200],
                    "snippet": str(snippet or ""),
                    "dataset_name": str(dataset_name or ""),
                })
            return results
        except Exception as e:
            logger.debug(f"Evidence FTS search error: {e}")
            return []

    def close(self):
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass


# Singleton medical knowledge backend
_medical_backend: Optional[MedicalKnowledgeBackend] = None
_medical_lock = threading.Lock()


def get_medical_backend(db_path: Optional[str] = None) -> MedicalKnowledgeBackend:
    """Get or create the shared medical knowledge backend."""
    global _medical_backend
    if _medical_backend is None:
        with _medical_lock:
            if _medical_backend is None:
                _medical_backend = MedicalKnowledgeBackend(db_path=db_path)
    return _medical_backend


class WikiSearchBackend:
    """Backend for searching the offline Wikipedia dump using FTS5.

    This wraps the snapshot-po wiki_dump_tools for use within BIOAgents.
    Supports:
    - FTS5 full-text search via SQLite
    - Page browsing via JSONL + offset index
    """

    def __init__(
        self,
        wiki_root: Optional[str] = None,
        search_topk: int = 8,
        browse_max_chars: int = 6000,
    ):
        self._wiki_root = wiki_root
        self._search_topk = search_topk
        self._browse_max_chars = browse_max_chars
        self._initialized = False
        self._search_tool = None
        self._browse_tool = None
        self._lock = threading.Lock()

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of the wiki backend."""
        if self._initialized:
            return self._search_tool is not None

        with self._lock:
            if self._initialized:
                return self._search_tool is not None

            try:
                wiki_root = self._resolve_wiki_root()
                if wiki_root is None:
                    logger.debug("No wiki dump found, wiki search disabled")
                    self._initialized = True
                    return False

                import sqlite3

                # Check required files
                pages_jsonl = os.path.join(wiki_root, "wiki_webpages.jsonl")
                pages_index = os.path.join(wiki_root, "wiki_pages_offset.sqlite")
                corpus_fts = os.path.join(wiki_root, "wiki_corpus_fts.sqlite")

                if not all(os.path.exists(p) for p in [pages_jsonl, pages_index, corpus_fts]):
                    logger.debug(f"Wiki dump at {wiki_root} is incomplete")
                    self._initialized = True
                    return False

                # Initialize lightweight FTS search
                self._fts_conn = sqlite3.connect(corpus_fts)
                self._fts_conn.execute("PRAGMA journal_mode=WAL;")

                # Initialize page accessor
                self._pages_conn = sqlite3.connect(pages_index)
                self._pages_fh = open(pages_jsonl, "rb")

                self._wiki_root = wiki_root
                self._initialized = True
                self._search_tool = True  # flag that search is available
                logger.info(f"Wiki search backend initialized: {wiki_root}")
                return True

            except Exception as e:
                logger.warning(f"Failed to initialize wiki backend: {e}")
                self._initialized = True
                return False

    def _resolve_wiki_root(self) -> Optional[str]:
        """Find the wiki dump root directory."""
        # 1. Explicit path
        if self._wiki_root and os.path.isdir(self._wiki_root):
            return self._wiki_root

        # 2. Environment variable
        env_root = os.environ.get("WIKI_ROOT")
        if env_root and os.path.isdir(env_root):
            return env_root

        # 3. Search common locations
        search_dirs = [
            # Direct workspace path
            "/data/project/private/minstar/workspace/wiki2018",
            # BIOAgents databases
            os.path.join(os.path.dirname(__file__), "../../databases/wiki2018_en"),
            os.path.join(os.path.dirname(__file__), "../../databases/wiki2026_en"),
        ]

        for d in search_dirs:
            d = os.path.abspath(d)
            if os.path.isdir(d):
                # Check for required artifacts
                needed = ["wiki_webpages.jsonl", "wiki_pages_offset.sqlite", "wiki_corpus_fts.sqlite"]
                if all(os.path.exists(os.path.join(d, f)) for f in needed):
                    return d

        return None

    def search(self, queries: List[str], topk: int = 0) -> List[Dict[str, str]]:
        """Search Wikipedia using FTS5."""
        if not self._ensure_initialized():
            return []

        topk = topk or self._search_topk
        results = []

        for query in queries[:5]:  # Limit to 5 queries
            query = (query or "").strip()
            if not query:
                continue

            # Extract search terms
            terms = [t.lower() for t in re.findall(r"[A-Za-z0-9]{2,}", query)]
            stop_words = {
                "what", "which", "who", "when", "where", "why", "how",
                "is", "are", "was", "were", "the", "a", "an", "and", "or",
                "of", "to", "in", "on", "for", "with", "by", "from",
            }
            filtered = [t for t in terms if t not in stop_words][:10]
            if not filtered:
                filtered = terms[:8]

            fts_query = " OR ".join(filtered)
            if not fts_query:
                continue

            try:
                cur = self._fts_conn.execute(
                    "SELECT url, title, snippet(corpus_fts, 2, '<b>', '</b>', '...', 16) AS sn "
                    "FROM corpus_fts WHERE corpus_fts MATCH ? ORDER BY bm25(corpus_fts) LIMIT ?",
                    (fts_query, topk),
                )
                for url, title, snippet in cur.fetchall():
                    results.append({
                        "title": str(title or "").strip(),
                        "url": str(url or "").strip(),
                        "snippet": str(snippet or "").strip(),
                        "source": "wikipedia",
                    })
            except Exception as e:
                logger.debug(f"Wiki FTS search error: {e}")

        return results

    def browse(self, url: str, query: str = "") -> str:
        """Browse a Wikipedia page by URL."""
        if not self._ensure_initialized():
            return "Wiki browse unavailable"

        # Normalize URL
        u = (url or "").strip()
        if "index.php/" in u:
            u = u.replace("index.php/", "index.php?title=")
        if "/wiki/" in u:
            u = u.replace("/wiki/", "/w/index.php?title=")
        if "_" in u:
            u = u.replace("_", "%20")

        try:
            cur = self._pages_conn.execute("SELECT offset FROM pages WHERE url=?", (u,))
            row = cur.fetchone()
            if not row:
                return f"Page not found: {url}"

            self._pages_fh.seek(int(row[0]))
            line = self._pages_fh.readline()
            obj = json.loads(line)
            text = str(obj.get("contents", "")).strip()

            if not text:
                return f"Empty page: {url}"

            # Extract relevant lines based on query
            if query:
                q_terms = set(re.findall(r"[A-Za-z0-9]{2,}", query.lower()))
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                scored = []
                for i, ln in enumerate(lines):
                    toks = set(re.findall(r"[A-Za-z0-9]{2,}", ln.lower()))
                    score = len(toks & q_terms) * 3
                    scored.append((score, i, ln))
                scored.sort(reverse=True)

                # Take top lines + lead
                lead = lines[:20]
                top_lines = [ln for _, _, ln in scored[:30] if _ > 0]
                seen = set()
                out = []
                for ln in lead + top_lines:
                    if ln not in seen:
                        seen.add(ln)
                        out.append(ln)
                text = "\n".join(out)

            return text[:self._browse_max_chars]

        except Exception as e:
            return f"Browse error: {e}"

    def close(self):
        """Clean up resources."""
        try:
            if hasattr(self, "_fts_conn"):
                self._fts_conn.close()
        except Exception:
            pass
        try:
            if hasattr(self, "_pages_conn"):
                self._pages_conn.close()
        except Exception:
            pass
        try:
            if hasattr(self, "_pages_fh"):
                self._pages_fh.close()
        except Exception:
            pass


# Singleton wiki backend (shared across all KnowledgeTools instances)
_wiki_backend: Optional[WikiSearchBackend] = None
_wiki_lock = threading.Lock()


def get_wiki_backend(wiki_root: Optional[str] = None) -> WikiSearchBackend:
    """Get or create the shared wiki search backend."""
    global _wiki_backend
    if _wiki_backend is None:
        with _wiki_lock:
            if _wiki_backend is None:
                _wiki_backend = WikiSearchBackend(wiki_root=wiki_root)
    return _wiki_backend


# ============================================================
# Unified Knowledge Tools
# ============================================================


class KnowledgeTools(ToolKitBase):
    """Unified knowledge search and browse tool set.

    This consolidates all search/browse tools into a single tool set that
    can be composed into any domain toolkit. Instead of each domain
    maintaining its own search tools, they all use this unified interface.

    The unified `search` tool routes queries across all available
    knowledge sources (PubMed, Wiki, Evidence, Guidelines, Wikipedia dump)
    and returns merged, ranked results.

    Usage:
        # As a standalone toolkit
        tools = KnowledgeTools(db=medical_qa_db)
        results = tools.search(queries=["cisplatin mechanism"])

        # As part of a domain toolkit (composition)
        class ClinicalTools(ToolKitBase):
            def __init__(self, db, knowledge_tools):
                self.knowledge = knowledge_tools
                # domain-specific tools...
    """

    def __init__(
        self,
        db=None,
        wiki_root: Optional[str] = None,
        enable_wiki: bool = True,
        enable_medical_kb: bool = True,
        medical_fts_path: Optional[str] = None,
    ):
        super().__init__(db)
        self._wiki_backend = get_wiki_backend(wiki_root) if enable_wiki else None
        self._medical_backend = get_medical_backend(medical_fts_path) if enable_medical_kb else None

    # ==========================================
    # Unified Search (routes across all sources)
    # ==========================================

    @is_tool(ToolType.READ)
    def search(self, queries: str, max_results: int = 8) -> list:
        """Search across all available medical knowledge sources.

        Searches PubMed literature, medical encyclopedia, evidence passages,
        clinical guidelines, and Wikipedia simultaneously. Returns merged
        results ranked by relevance.

        Args:
            queries: Search queries, comma-separated for multiple (e.g., 'cisplatin mechanism, hearing loss treatment')
            max_results: Maximum total results to return (default 8)

        Returns:
            List of search results from all sources, ranked by relevance
        """
        max_results = int(max_results)
        query_list = [q.strip() for q in queries.split(",") if q.strip()]
        if not query_list:
            return [{"message": "No queries provided."}]

        all_results = []

        # 1. Search PubMed articles (from domain DB)
        if self.db and hasattr(self.db, "articles"):
            for query in query_list:
                results = self._search_articles(query, max_per_source=3)
                all_results.extend(results)

        # 2. Search Medical Wiki (from domain DB)
        if self.db and hasattr(self.db, "wiki_entries"):
            for query in query_list:
                results = self._search_wiki_entries(query, max_per_source=3)
                all_results.extend(results)

        # 3. Search Evidence Passages (from domain DB)
        if self.db and hasattr(self.db, "evidence_passages"):
            for query in query_list:
                results = self._search_evidence(query, max_per_source=3)
                all_results.extend(results)

        # 4. Search medical knowledge FTS (MedCPT, PubMed, PMC, textbooks)
        if self._medical_backend:
            for query in query_list:
                med_results = self._medical_backend.search_passages(query, topk=4)
                for mr in med_results:
                    all_results.append({
                        "source": mr.get("source", "medical_kb"),
                        "title": mr.get("title", ""),
                        "snippet": mr.get("snippet", ""),
                        "doc_id": mr.get("doc_id", ""),
                        "category": mr.get("category", ""),
                        "relevance": 0.7,  # Higher default for medical-specific
                    })

        # 5. Search Wikipedia dump (FTS5)
        if self._wiki_backend:
            wiki_results = self._wiki_backend.search(query_list, topk=4)
            for wr in wiki_results:
                all_results.append({
                    "source": "wikipedia",
                    "title": wr.get("title", ""),
                    "snippet": wr.get("snippet", ""),
                    "url": wr.get("url", ""),
                    "relevance": 0.5,  # Default relevance for wiki results
                })

        # Sort by relevance and deduplicate
        all_results.sort(key=lambda x: x.get("relevance", 0), reverse=True)

        # Deduplicate by title
        seen_titles = set()
        unique_results = []
        for r in all_results:
            title_key = r.get("title", "").lower().strip()
            if title_key and title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            unique_results.append(r)

        if not unique_results:
            return [{"message": f"No results found for: {queries}"}]

        return unique_results[:max_results]

    @is_tool(ToolType.READ)
    def browse(self, url_or_id: str, query: str = "") -> dict:
        """Browse a specific knowledge source by URL or ID.

        Can browse PubMed articles (by PMID), wiki entries (by entry_id),
        evidence passages (by passage_id), or Wikipedia pages (by URL).

        Args:
            url_or_id: PMID, entry ID, passage ID, or Wikipedia URL
            query: Optional query to highlight relevant sections

        Returns:
            Full content of the browsed resource
        """
        url_or_id = str(url_or_id).strip()

        # Try PubMed article (PMID format)
        if self.db and hasattr(self.db, "articles"):
            if url_or_id in self.db.articles:
                article = self.db.articles[url_or_id]
                result = {
                    "source": "pubmed",
                    "pmid": getattr(article, "pmid", url_or_id),
                    "title": getattr(article, "title", ""),
                    "abstract": getattr(article, "abstract", ""),
                    "authors": getattr(article, "authors", []),
                    "journal": getattr(article, "journal", ""),
                    "year": getattr(article, "year", ""),
                }
                if hasattr(article, "sections") and article.sections:
                    result["available_sections"] = list(article.sections.keys())
                return result

        # Try wiki entry
        if self.db and hasattr(self.db, "wiki_entries"):
            if url_or_id in self.db.wiki_entries:
                entry = self.db.wiki_entries[url_or_id]
                return {
                    "source": "medical_wiki",
                    "entry_id": getattr(entry, "entry_id", url_or_id),
                    "title": getattr(entry, "title", ""),
                    "summary": getattr(entry, "summary", ""),
                    "categories": getattr(entry, "categories", []),
                    "full_text": getattr(entry, "full_text", ""),
                }

        # Try evidence passage
        if self.db and hasattr(self.db, "evidence_passages"):
            if url_or_id in self.db.evidence_passages:
                passage = self.db.evidence_passages[url_or_id]
                return {
                    "source": "evidence",
                    "passage_id": getattr(passage, "passage_id", url_or_id),
                    "title": getattr(passage, "title", ""),
                    "text": getattr(passage, "text", ""),
                    "category": getattr(passage, "category", ""),
                }

        # Try Wikipedia page (URL)
        if self._wiki_backend and ("wiki" in url_or_id.lower() or "/" in url_or_id):
            content = self._wiki_backend.browse(url_or_id, query=query)
            return {
                "source": "wikipedia",
                "url": url_or_id,
                "content": content,
            }

        return {"error": f"Resource '{url_or_id}' not found in any knowledge source."}

    # ==========================================
    # Source-Specific Search Tools
    # ==========================================

    @is_tool(ToolType.READ)
    def search_pubmed(self, query: str, max_results: int = 5) -> list:
        """Search PubMed-style medical literature for articles.

        Args:
            query: Search query (e.g., 'cisplatin ototoxicity mechanism')
            max_results: Maximum number of results (default 5)

        Returns:
            List of matching articles with title, abstract snippet, and PMID
        """
        if not self.db or not hasattr(self.db, "articles"):
            # Fallback to wiki search
            if self._wiki_backend:
                results = self._wiki_backend.search([query], topk=max_results)
                return [{"source": "wikipedia", **r} for r in results]
            return [{"message": "No literature database available."}]

        return self._search_articles(query, max_per_source=int(max_results))

    @is_tool(ToolType.READ)
    def search_medical_wiki(self, query: str, max_results: int = 5) -> list:
        """Search the medical encyclopedia / wiki for entries.

        Args:
            query: Search query (e.g., 'sensorineural hearing loss')
            max_results: Maximum number of results (default 5)

        Returns:
            List of matching encyclopedia entries
        """
        results = []

        # Search domain wiki entries
        if self.db and hasattr(self.db, "wiki_entries"):
            results.extend(self._search_wiki_entries(query, max_per_source=int(max_results)))

        # Supplement with Wikipedia dump
        if self._wiki_backend and len(results) < max_results:
            remaining = max_results - len(results)
            wiki_results = self._wiki_backend.search([query], topk=remaining)
            for wr in wiki_results:
                results.append({
                    "source": "wikipedia",
                    "title": wr.get("title", ""),
                    "snippet": wr.get("snippet", ""),
                    "url": wr.get("url", ""),
                    "relevance": 0.5,
                })

        if not results:
            return [{"message": f"No entries found for '{query}'."}]
        return results[:max_results]

    @is_tool(ToolType.READ)
    def search_evidence(self, query: str, max_results: int = 5, category: str = "") -> list:
        """Retrieve relevant evidence passages from medical textbooks, PubMed, and PMC literature.

        Searches across MedCPT evidence (581K PubMed/PMC passages), biomedical
        instruction knowledge, and domain-specific evidence passages.

        Args:
            query: The medical question or topic
            max_results: Maximum number of passages (default 5)
            category: Optional category filter (e.g., 'pharmacology', 'pathology')

        Returns:
            List of evidence passages ranked by relevance
        """
        results = []

        # 1. Search medical knowledge FTS (MedCPT + biomedical instructions)
        if self._medical_backend:
            med_results = self._medical_backend.search_evidence(query, topk=max_results)
            for mr in med_results:
                results.append({
                    "source": mr.get("source", "medcpt_evidence"),
                    "doc_id": mr.get("doc_id", ""),
                    "title": mr.get("question", "")[:120],
                    "snippet": mr.get("snippet", ""),
                    "dataset_name": mr.get("dataset_name", ""),
                    "relevance": 0.8,
                })

        # 2. Search domain DB evidence
        if self.db and hasattr(self.db, "evidence_passages"):
            db_results = self._search_evidence(query, max_per_source=max_results, category=category)
            results.extend(db_results)

        # Deduplicate and sort
        results.sort(key=lambda x: x.get("relevance", 0), reverse=True)

        if not results:
            # Fallback to wiki
            return self.search_medical_wiki(query, max_results=max_results)

        return results[:max_results]

    @is_tool(ToolType.READ)
    def search_guidelines(self, condition: str) -> list:
        """Search clinical guidelines for a specific condition.

        Args:
            condition: The medical condition (e.g., 'hypertension', 'diabetes type 2')

        Returns:
            List of relevant clinical guidelines with source and recommendations
        """
        results = []

        # Search domain guidelines
        if self.db and hasattr(self.db, "guidelines"):
            for gid, guideline in self.db.guidelines.items():
                combined = f"{getattr(guideline, 'condition', '')} {getattr(guideline, 'title', '')} {getattr(guideline, 'summary', '')}"
                score = self._relevance_score(condition, combined)
                if score > 0:
                    results.append({
                        "source": "clinical_guidelines",
                        "guideline_id": gid,
                        "title": getattr(guideline, "title", ""),
                        "condition": getattr(guideline, "condition", ""),
                        "summary": getattr(guideline, "summary", "")[:300],
                        "relevance": round(score, 3),
                    })

        # Also search evidence and wiki for guidelines-related content
        if self.db and hasattr(self.db, "evidence_passages"):
            guideline_query = f"clinical guidelines {condition} treatment recommendation"
            evidence = self._search_evidence(guideline_query, max_per_source=3)
            results.extend(evidence)

        results.sort(key=lambda x: x.get("relevance", 0), reverse=True)

        if not results:
            return [{"message": f"No guidelines found for '{condition}'."}]
        return results[:5]

    @is_tool(ToolType.READ)
    def browse_article(self, pmid: str, section: str = "") -> dict:
        """Browse a specific article by PMID, optionally reading a specific section.

        Args:
            pmid: The PubMed ID of the article
            section: Optional section to read (e.g., 'methods', 'results')

        Returns:
            Article details including title, abstract, and sections
        """
        return self.browse(pmid, query=section)

    @is_tool(ToolType.READ)
    def browse_wiki_entry(self, entry_id: str) -> dict:
        """Browse a specific medical encyclopedia entry by ID.

        Args:
            entry_id: The unique identifier of the wiki entry

        Returns:
            Full entry content
        """
        return self.browse(entry_id)

    # ==========================================
    # Internal helpers
    # ==========================================

    def _relevance_score(self, query: str, text: str) -> float:
        """Compute simple keyword-overlap relevance score."""
        query_tokens = set(re.findall(r"\w+", query.lower()))
        text_tokens = set(re.findall(r"\w+", text.lower()))
        if not query_tokens:
            return 0.0
        overlap = query_tokens & text_tokens
        return len(overlap) / len(query_tokens)

    def _search_articles(self, query: str, max_per_source: int = 5) -> list:
        """Search PubMed-style articles from db."""
        if not self.db or not hasattr(self.db, "articles"):
            return []

        scored = []
        for pmid, article in self.db.articles.items():
            combined = f"{getattr(article, 'title', '')} {getattr(article, 'abstract', '')} {' '.join(getattr(article, 'keywords', []))}"
            score = self._relevance_score(query, combined)
            if score > 0:
                scored.append((score, pmid, article))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, pmid, article in scored[:max_per_source]:
            abstract = getattr(article, "abstract", "")
            snippet = abstract[:300] + ("..." if len(abstract) > 300 else "")
            results.append({
                "source": "pubmed",
                "pmid": pmid,
                "title": getattr(article, "title", ""),
                "snippet": snippet,
                "journal": getattr(article, "journal", ""),
                "year": getattr(article, "year", ""),
                "relevance": round(score, 3),
            })
        return results

    def _search_wiki_entries(self, query: str, max_per_source: int = 5) -> list:
        """Search medical wiki entries from db."""
        if not self.db or not hasattr(self.db, "wiki_entries"):
            return []

        scored = []
        for eid, entry in self.db.wiki_entries.items():
            combined = f"{getattr(entry, 'title', '')} {getattr(entry, 'summary', '')} {' '.join(getattr(entry, 'categories', []))}"
            score = self._relevance_score(query, combined)
            if score > 0:
                scored.append((score, eid, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, eid, entry in scored[:max_per_source]:
            summary = getattr(entry, "summary", "")
            snippet = summary[:250] + ("..." if len(summary) > 250 else "")
            results.append({
                "source": "medical_wiki",
                "entry_id": eid,
                "title": getattr(entry, "title", ""),
                "snippet": snippet,
                "categories": getattr(entry, "categories", []),
                "relevance": round(score, 3),
            })
        return results

    def _search_evidence(
        self, query: str, max_per_source: int = 5, category: str = "",
    ) -> list:
        """Search evidence passages from db."""
        if not self.db or not hasattr(self.db, "evidence_passages"):
            return []

        candidates = self.db.evidence_passages.values()
        if category:
            candidates = [
                p for p in candidates
                if getattr(p, "category", "").lower() == category.lower()
            ]

        scored = []
        for passage in candidates:
            combined = f"{getattr(passage, 'title', '')} {getattr(passage, 'text', '')}"
            score = self._relevance_score(query, combined)
            if score > 0:
                scored.append((score, passage))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, passage in scored[:max_per_source]:
            text = getattr(passage, "text", "")
            snippet = text[:400] + ("..." if len(text) > 400 else "")
            results.append({
                "source": "evidence",
                "passage_id": getattr(passage, "passage_id", ""),
                "title": getattr(passage, "title", ""),
                "snippet": snippet,
                "category": getattr(passage, "category", ""),
                "relevance": round(score, 3),
            })
        return results

    # ==========================================
    # Reasoning / Think / Submit
    # ==========================================

    @is_tool(ToolType.GENERIC)
    def think(self, thought: str) -> str:
        """Internal reasoning tool. Use to think through complex problems.

        Args:
            thought: Your reasoning process

        Returns:
            Empty string (thinking is logged internally)
        """
        return ""

    @is_tool(ToolType.GENERIC)
    def submit_answer(self, answer: str, reasoning: str = "") -> str:
        """Submit your final answer.

        Args:
            answer: The answer (e.g., 'A', 'B', or free-text)
            reasoning: Your reasoning for the answer

        Returns:
            Confirmation of submission
        """
        return f"Answer '{answer}' submitted. Reasoning: {reasoning}"
