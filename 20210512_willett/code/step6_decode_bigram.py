# -*- coding: utf-8 -*-
"""M1b-B：纯 Python 大词表 bigram 解码器（惰性组合 HCL × G，复刻 latgen-lazylm 语义）。

为什么惰性组合：离线 fstcompose HCL(45 万弧)×G(1030 万弧) 的积图是内存炸弹（实测 WSL 吃满 47GB）。
官方用 latgen-lazylm-faster-mapped 就是为了惰性组合——本解码器同样在 token 态上按需展开乘积空间。

输入：
- decode_cache/HCL.txt、decode_cache/G.txt（openfst fstprint 文本，G 需先按 ilabel 排序亦可，此处重排）
- BigramLM/model（文本版 TransitionModel：tid→pdf、tid→phone、tid 转移 log 概率）
- Step6 KaldiMatrices/kaldiMat_*.txt

解码语义（对照 kaldi DecodableMatrixScaledMapped + latgen 参数）：
- HCL 发射弧（ilabel=tid>0）消耗一帧：声学代价 += 1.79 * -(mat[f,pdf(tid)] + tlp[tid])
- 图代价 += HCL 弧权 + G 弧权（含 backoff ε 路径代价）；热带半环自然对数
- HCL ol>0 的弧推进 G：从当前 G 态经 ε-backoff 到某态后走 ilabel=ol 的弧
- beam=65（相对帧内最优）；终态 = HCL 终态 ∧ G 终态，补两图 final
- n-best 128 从 token 网格 A* 枚举；选句 argmin(acCost + 2.0*graphCost)

transcript 重建：路径发射弧 phone 序列 → 游程折叠 → 过滤 '.'（词典消歧哑符号）→
<space>→' ' → 去首尾空格（<ctc> 丢弃）。已对包内 best_ali 样例人工核验。

用法：
    python step6_decode_bigram.py --cvPart HeldOutTrials --limit 5 --sessions t5.2019.05.08
    python step6_decode_bigram.py --cvPart HeldOutTrials --limit 20
"""
import argparse
import heapq
import os
import re
import sys

import numpy as np
import scipy.io

REPO = r'.\20210512_willett'
ROOT = os.path.join(REPO, 'data', '20210512_willett', 'handwritingBCIData')
LM_DIR = os.path.join(ROOT, 'BigramLM')
CACHE = os.path.join(REPO, 'data', '20210512_willett', 'decode_cache')

SESSIONS = ['t5.2019.05.08', 't5.2019.11.25', 't5.2019.12.09', 't5.2019.12.11', 't5.2019.12.18',
            't5.2019.12.20', 't5.2020.01.06', 't5.2020.01.08', 't5.2020.01.13', 't5.2020.01.15']

ACOUSTIC_SCALE = 1.79
BEAM = 65.0
NBEST = 128


def load_model(path):
    """Parse the text TransitionModel into per-tid lookup arrays.

    Reads the Kaldi transition-model text file, expands each (phone,
    hmm-state, pdf) triple into its tid block (2 tids for the ``<ctc>``
    phone 33, otherwise 3 tids for state0 and 2 for state1) and collects
    the per-tid transition log-probabilities.

    Parameters
    ----------
    path : str
        Path to the text ``TransitionModel`` file (``BigramLM/model``).

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray)
        ``(tid2pdf, tid2phone, tid_lp)``: int32 arrays indexed by tid giving
        the pdf-id, phone-id and transition log-probability respectively
        (index 0 is unused padding).

    Raises
    ------
    AssertionError
        If the parsed triple count or expanded tid count is inconsistent
        with the file's declared totals.
    """
    txt = open(path).read()
    n_triples = int(re.search(r'<Triples>\s*(\d+)', txt).group(1))
    body = txt.split('<Triples>')[1].split('</Triples>')[0]
    rows = [tuple(map(int, l.split())) for l in body.strip().split('\n') if l.strip()]
    rows = [r for r in rows if len(r) == 3]  # 首行是计数
    assert len(rows) == n_triples
    lp = np.array([float(x) for x in re.search(r'<LogProbs>\s*\[(.*?)\]', txt, re.S).group(1).split()])
    tid2pdf = np.zeros(len(lp), dtype=np.int32)
    tid2phone = np.zeros(len(lp), dtype=np.int32)
    tid_lp = np.zeros(len(lp), dtype=np.float64)
    t = 1
    for phone, hstate, pdf in rows:
        n = 2 if phone == 33 else (3 if hstate == 0 else 2)
        for _ in range(n):
            tid2pdf[t] = pdf
            tid2phone[t] = phone
            tid_lp[t] = lp[t]
            t += 1
    assert t == len(lp)
    return tid2pdf, tid2phone, tid_lp


def load_symbols(path):
    """Load a Kaldi symbol table (symbol -> id) from a text file.

    Parameters
    ----------
    path : str
        Path to a symbols file such as ``BigramLM/chars.txt`` with
        ``symbol id`` per line.

    Returns
    -------
    dict
        Mapping from symbol string to integer id; malformed lines (not
        exactly two fields) are skipped.
    """
    m = {}
    for line in open(path, encoding='utf-8'):
        parts = line.split()
        if len(parts) == 2:
            m[parts[0]] = int(parts[1])
    return m


def parse_fst_text(path, name):
    """AT&T 文本 → (arc_src, arc_dst, arc_il, arc_ol, arc_w, offset, finals, n_states)。
    pandas C 引擎整体读入（10M 行约 1-2 分钟、~1GB 内存），终态行（字段数≤2）单独抽出。

    Parameters
    ----------
    path : str
        Path to an OpenFst ``fstprint`` AT&T-style text file.
    name : str
        Display name of the graph (used in progress logging only).

    Returns
    -------
    tuple
        ``(src, dst, il, ol, w, finals, n_states)`` where ``src``/``dst``/
        ``il``/``ol`` are int32 arc arrays, ``w`` the float64 arc weights,
        ``finals`` a dict of final state -> weight, and ``n_states`` the
        number of states.
    """
    import pandas as pd
    print(f'parsing {name}: {path}', flush=True)
    df = pd.read_csv(path, sep=r'\s+', header=None, dtype=np.float64,
                     names=['s', 'd', 'il', 'ol', 'w'], usecols=[0, 1, 2, 3, 4],
                     keep_default_na=False, na_values=[''])
    n_states = int(max(df['s'].max(), df['d'].max())) + 1
    is_arc = df['il'].notna()
    src = df.loc[is_arc, 's'].to_numpy(dtype=np.int32)
    dst = df.loc[is_arc, 'd'].to_numpy(dtype=np.int32)
    il = df.loc[is_arc, 'il'].to_numpy(dtype=np.int32)
    ol = df.loc[is_arc, 'ol'].to_numpy(dtype=np.int32)
    w = df.loc[is_arc, 'w'].fillna(0.0).to_numpy(dtype=np.float64)
    finals = {}
    fin_rows = df.loc[~is_arc]
    for s, wv in zip(fin_rows['s'].to_numpy(dtype=np.int32), fin_rows['d']):
        finals[int(s)] = float(wv) if not pd.isna(wv) else 0.0
    del df
    src = np.asarray(src, dtype=np.int32)
    dst = np.asarray(dst, dtype=np.int32)
    il = np.asarray(il, dtype=np.int32)
    ol = np.asarray(ol, dtype=np.int32)
    w = np.asarray(w, dtype=np.float64)
    print(f'  {name}: {n_states} states, {len(src)} arcs, {len(finals)} finals', flush=True)
    return src, dst, il, ol, w, finals, n_states


class Graph:
    """按 (state, ilabel) 排好序的 FST，支持切片访问与 ε 闭包。

    Arc arrays are lexsorted by (source state, ilabel) so every state's arcs
    form one contiguous, ilabel-ascending segment addressable via
    :meth:`arcs`, with epsilon (ilabel==0) arcs first within each segment.

    Parameters
    ----------
    path : str
        Path to the AT&T text FST file parsed with ``parse_fst_text``.
    name : str
        Display name of the graph (``'HCL'`` or ``'G'``), used for logging.
    """

    def __init__(self, path, name):
        """Parse the FST text and build the CSR-style arc index.

        Parameters
        ----------
        path : str
            Path to the AT&T text FST file.
        name : str
            Display name of the graph, used for logging.

        Returns
        -------
        None
            Populates ``self.src/dst/il/ol/w``, the per-state ``offset``
            table, ``self.finals``, ``self.n_states`` and ``self.start=0``.
        """
        src, dst, il, ol, w, finals, n_states = parse_fst_text(path, name)
        order = np.lexsort((il, src))  # 先按 src 再按 il
        self.src, self.dst, self.il, self.ol, self.w = src[order], dst[order], il[order], ol[order], w[order]
        counts = np.zeros(n_states + 1, dtype=np.int64)
        np.add.at(counts, self.src, 1)
        self.offset = np.zeros(n_states + 1, dtype=np.int64)
        self.offset[1:] = np.cumsum(counts)[:n_states]
        self.finals = finals
        self.n_states = n_states
        self.start = 0
        # 每个 state 的弧是否已按 ilabel 分好段：ilabel==0 的弧在最前（lexsort 保证）
        self._closure_cache = {}

    def arcs(self, s):
        """Return the contiguous arc-array slice of state ``s``.

        Parameters
        ----------
        s : int
            State id.

        Returns
        -------
        slice
            Index range ``[offset[s], offset[s+1])`` into the arc arrays.
        """
        return slice(self.offset[s], self.offset[s + 1])

    def word_arcs(self, s, word):
        """state s 上 ilabel==word 的弧：段内二分。

        Parameters
        ----------
        s : int
            State id whose arc segment is searched.
        word : int
            Ilabel (output word of HCL) to locate.

        Returns
        -------
        tuple of (int, int) or None
            Half-open index range ``(start, stop)`` of matching arcs, or
            None when the state has no arc with that ilabel.
        """
        sl = self.arcs(s)
        los, his = sl.start, sl.stop
        # ilabel 在段内升序
        lo, hi = los, his
        while lo < hi:
            mid = (lo + hi) // 2
            if self.il[mid] < word:
                lo = mid + 1
            else:
                hi = mid
        start = lo
        hi = his
        while lo < hi:
            mid = (lo + hi) // 2
            if self.il[mid] <= word:
                lo = mid + 1
            else:
                hi = mid
        if start >= lo:
            return None
        return start, lo

    def eps_closure(self, s):
        """ε(ilabel==0) 闭包：返回 [(g', 累计代价)]，含自身 (s, 0)。

        Parameters
        ----------
        s : int
            Starting state of the closure.

        Returns
        -------
        list of (int, float)
            ``(state, accumulated epsilon-arc cost)`` pairs reachable from
            ``s`` via ilabel==0 arcs, including ``(s, 0.0)`` itself. Results
            are memoised in ``self._closure_cache``.
        """
        c = self._closure_cache.get(s)
        if c is not None:
            return c
        out = [(s, 0.0)]
        seen = {s}
        frontier = [(s, 0.0)]
        while frontier:
            g, cost = frontier.pop()
            sl = self.arcs(g)
            for i in range(sl.start, sl.stop):
                if self.il[i] != 0:
                    break  # il 升序，遇到非 0 即止
                d, w = int(self.dst[i]), self.w[i]
                if d not in seen:
                    seen.add(d)
                    out.append((d, cost + w))
                    frontier.append((d, cost + w))
        self._closure_cache[s] = out
        return out


class LazyDecoder:
    """token = (hcl_state, g_state)；HCL 弧 ol>0 时推进 G。

    Time-synchronous beam search over the lazily composed HCL x G product:
    HCL arcs with a nonzero output label expand the G grammar through its
    epsilon-backoff closure, reproducing Kaldi's latgen-lazylm semantics.

    Parameters
    ----------
    hcl : Graph
        The HCL graph (arcs with ilabel>0 are emitting, consuming a frame).
    g : Graph
        The bigram grammar G graph.
    tid2pdf : numpy.ndarray
        Per-tid pdf-id lookup from ``load_model``.
    tid_lp : numpy.ndarray
        Per-tid transition log-probability lookup from ``load_model``.
    """

    def __init__(self, hcl, g, tid2pdf, tid_lp):
        """Precompute per-HCL-arc static acoustic costs and pdf indices.

        Parameters
        ----------
        hcl : Graph
            The HCL graph.
        g : Graph
            The bigram grammar G graph (stored for ``decode``).
        tid2pdf : numpy.ndarray
            Per-tid pdf-id lookup from ``load_model``.
        tid_lp : numpy.ndarray
            Per-tid transition log-probability lookup from ``load_model``.

        Returns
        -------
        None
            Populates ``self.arc_ac_static`` (transition-model part of the
            acoustic cost, ACOUSTIC_SCALE * -tid_lp) and ``self.arc_pdf``
            (pdf index per arc, 0 for non-emitting arcs).
        """
        self.hcl, self.g = hcl, g
        self.arc_ac_static = np.where(hcl.il > 0,
                                      ACOUSTIC_SCALE * (-tid_lp[np.minimum(hcl.il, len(tid_lp) - 1)]), 0.0)
        self.arc_pdf = tid2pdf[np.minimum(hcl.il, len(tid2pdf) - 1)].copy()
        self.arc_pdf[hcl.il == 0] = 0

    def decode(self, mat, nbest=NBEST):
        """Beam-search decode one sentence and enumerate n-best paths.

        Runs the frame-synchronous token pass (beam=65 relative to the
        running best), then a backward Dijkstra over the resulting lattice
        for admissible A* heuristics, and finally A* enumeration of up to
        ``nbest`` complete paths.

        Parameters
        ----------
        mat : numpy.ndarray
            ``(T, 32)`` log-likelihood matrix (rows = frames, cols = pdfs)
            from a kaldiMat file.
        nbest : int, optional
            Maximum number of best paths to enumerate (default 128).

        Returns
        -------
        list of tuple
            Up to ``nbest`` entries ``(ac_cost, graph_cost, phones,
            words)``: acoustic and graph costs (final weights included) of
            each path plus its emitted phone-id and word-id sequences.
            Empty list when no token survives to a jointly final state.
        """
        hcl, g = self.hcl, self.g
        T = mat.shape[0]
        frame_cost = np.zeros(33, dtype=np.float64)
        # 节点: (ac, graph, parent, hcl_arc, g_arc_start, g_arc_stop, g_eps_cost, g_state_after)
        n_ac, n_graph, n_parent, n_harc = [], [], [], []
        n_garc_lo, n_garc_hi, n_geps, n_gnext = [], [], [], []
        toks = {(hcl.start, g.start): (0.0, -1)}

        for f in range(T):
            frame_cost[1:] = ACOUSTIC_SCALE * (-mat[f])
            fc = frame_cost[self.arc_pdf]
            new = {}
            best = np.inf
            for (hs, gs), (tot, node) in toks.items():
                pac, pgraph = (0.0, 0.0) if node < 0 else (n_ac[node], n_graph[node])
                sl = hcl.arcs(hs)
                if sl.start == sl.stop:
                    continue
                arc_ac = pac + self.arc_ac_static[sl] + fc[sl]
                arc_graph = pgraph + hcl.w[sl]
                arc_tot = arc_ac + arc_graph
                # 先只挑 tot 可进的弧，再逐弧推 G
                for k in range(sl.start, sl.stop):
                    nt = arc_tot[k - sl.start]
                    if best < np.inf and nt > best + BEAM:
                        continue
                    ol = int(hcl.ol[k])
                    if ol == 0:
                        key = (int(hcl.dst[k]), gs)
                        old = new.get(key)
                        if old is None or nt < old[0]:
                            idx = len(n_ac)
                            n_ac.append(arc_ac[k - sl.start]); n_graph.append(arc_graph[k - sl.start])
                            n_parent.append(node); n_harc.append(k)
                            n_garc_lo.append(-1); n_garc_hi.append(-1); n_geps.append(0.0); n_gnext.append(gs)
                            new[key] = (nt, idx)
                            if nt < best:
                                best = nt
                    else:
                        for gprime, eps_cost in g.eps_closure(gs):
                            wa = g.word_arcs(gprime, ol)
                            if wa is None:
                                continue
                            for ga in range(wa[0], wa[1]):
                                ntot = nt + eps_cost + g.w[ga]
                                if best < np.inf and ntot > best + BEAM:
                                    continue
                                key = (int(hcl.dst[k]), int(g.dst[ga]))
                                old = new.get(key)
                                if old is None or ntot < old[0]:
                                    idx = len(n_ac)
                                    n_ac.append(arc_ac[k - sl.start])
                                    n_graph.append(arc_graph[k - sl.start] + eps_cost + g.w[ga])
                                    n_parent.append(node); n_harc.append(k)
                                    n_garc_lo.append(wa[0]); n_garc_hi.append(wa[1]); n_geps.append(eps_cost)
                                    n_gnext.append(int(g.dst[ga]))
                                    new[key] = (ntot, idx)
                                    if ntot < best:
                                        best = ntot
            if not new:
                break
            toks = {s: v for s, v in new.items() if v[0] <= best + BEAM} or new

        ends = []
        for (hs, gs), (tot, node) in toks.items():
            hf = hcl.finals.get(hs)
            gf = g.finals.get(gs)
            if hf is not None and gf is not None:
                ends.append((tot + hf + gf, hs, gs, node))
        if not ends:
            return []
        ends.sort(key=lambda x: x[0])

        # 反向 Dijkstra（节点创建序 = 拓扑序，同帧内父先于子创建）
        n_nodes = len(n_ac)
        children = [[] for _ in range(n_nodes)]
        for i, p in enumerate(n_parent):
            if p >= 0:
                children[p].append(i)
        back = np.full(n_nodes, np.inf)
        heap = []
        for tot_f, hs, gs, node in ends:
            fin = hcl.finals[hs] + g.finals[gs]
            if back[node] > fin:
                back[node] = fin
        for i in range(n_nodes):
            if np.isfinite(back[i]):
                heapq.heappush(heap, (back[i], i))
        while heap:
            d, i = heapq.heappop(heap)
            if d > back[i]:
                continue
            p = n_parent[i]
            if p < 0:
                continue
            step = (n_ac[i] - n_ac[p]) + (n_graph[i] - n_graph[p])
            if d + step < back[p]:
                back[p] = d + step
                heapq.heappush(heap, (d + step, p))

        final_w = np.full(n_nodes, np.inf)
        # 终判定需 (hs, gs)：节点存了 harc（HCL 弧→dst 即 HCL 后态）与 gnext（G 后态）
        for i in range(n_nodes):
            hs = int(hcl.dst[n_harc[i]])
            gs = n_gnext[i]
            hf = hcl.finals.get(hs)
            gf = g.finals.get(gs)
            if hf is not None and gf is not None:
                final_w[i] = hf + gf

        roots = [i for i in range(n_nodes) if n_parent[i] == -1]
        # A* 用父指针链避免路径元组复制：搜索态 = (f, g, node, search_parent)
        astar = []
        seq = 0
        for r in roots:
            gg = n_ac[r] + n_graph[r]
            if np.isfinite(back[r]):
                heapq.heappush(astar, (gg + back[r], seq, gg, r, -1))
                seq += 1
        results = []
        pops = 0
        # 搜索态存储：search_node -> (search_parent, lattice_node)
        search_nodes = []
        while astar and len(results) < nbest and pops < 2_000_000:
            fsc, _, gg, node, sparent = heapq.heappop(astar)
            pops += 1
            my_search = len(search_nodes)
            search_nodes.append((sparent, node))
            if np.isfinite(final_w[node]):
                results.append((gg + final_w[node], n_ac[node], n_graph[node] + final_w[node], my_search))
                if len(results) >= nbest:
                    break
            for cch in children[node]:
                ng = n_ac[cch] + n_graph[cch]
                if np.isfinite(back[cch]):
                    heapq.heappush(astar, (ng + back[cch], seq, ng, cch, my_search))
                    seq += 1

        out = []
        for tot, ac, graph, search_idx in results:
            # 沿父指针链回溯网格节点
            chain = []
            si = search_idx
            while si >= 0:
                si, nd = search_nodes[si][0], search_nodes[si][1]
                chain.append(nd)
            chain.reverse()
            phones, words = [], []
            for nd in chain:
                arc = n_harc[nd]
                if hcl.il[arc] > 0:
                    phones.append(int(TID2PHONE[hcl.il[arc]]))
                if hcl.ol[arc] != 0:
                    words.append(int(hcl.ol[arc]))
            out.append((ac, graph, phones, words))
        return out


TID2PHONE = None


def phones_to_transcript(phones, chars_by_id):
    """Fold a phone-id path into a transcript string.

    Applies run-length folding (one character per contiguous phone run),
    drops disambiguation/blank symbols (``'.'``, ``<ctc>``, ``<eps>``),
    maps ``<space>`` to a space and strips leading/trailing spaces.

    Parameters
    ----------
    phones : list of int
        Emitted phone ids along a decode path.
    chars_by_id : dict
        Mapping from phone id to symbol string (inverse of chars.txt).

    Returns
    -------
    str
        The decoded transcript.
    """
    toks = []
    for ph in phones:
        if not toks or toks[-1] != ph:
            toks.append(ph)
    syms = []
    for ph in toks:
        s = chars_by_id.get(ph, '')
        if s in ('.', '<ctc>', '<eps>'):
            continue
        syms.append(' ' if s == '<space>' else s)
    return ''.join(syms).strip(' ')


def wer(r, h):
    """Levenshtein edit distance between two token sequences.

    Parameters
    ----------
    r : list or str
        Reference tokens (ground truth).
    h : list or str
        Hypothesis tokens (decoder output).

    Returns
    -------
    int
        Minimum number of insertions/deletions/substitutions to turn ``h``
        into ``r``.
    """
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    for i in range(len(r) + 1):
        d[i, 0] = i
    for j in range(len(h) + 1):
        d[0, j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = d[i - 1, j - 1] if r[i - 1] == h[j - 1] else min(d[i - 1, j - 1] + 1, d[i - 1, j] + 1, d[i, j - 1] + 1)
    return int(d[len(r), len(h)])


def load_kaldi_mat(path):
    """Load a kaldiMat text matrix into a float array.

    Skips the leading header line and the closing ``]`` line, parsing the
    remaining whitespace-separated numbers as matrix rows.

    Parameters
    ----------
    path : str
        Path to a ``kaldiMat_<sidx>.txt`` file.

    Returns
    -------
    numpy.ndarray
        ``(T, n_pdf)`` float64 matrix of frame log-likelihoods.
    """
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            s = line.strip()
            if not s or s == ']':
                continue
            rows.append([float(x) for x in s.split()])
    return np.asarray(rows)


def read_ref_selection(kout_dir, sidx):
    """包内 KaldiOutput 的官方选句文本（argmin(ac+2lm)）。

    Parameters
    ----------
    kout_dir : str
        Official ``KaldiOutput/<session>`` directory.
    sidx : int
        Sentence index within the session.

    Returns
    -------
    tuple or None
        ``(selected_text, all_texts, ac_scores, lm_scores)`` where the
        selected text is the n-best entry minimising ``ac + 2.0*lm``; None
        when the transcript or acoustic-score file is missing.
    """
    import math
    tr = os.path.join(kout_dir, f'{sidx}_transcript.txt')
    acf = os.path.join(kout_dir, f'{sidx}_best_acscore.ark')
    lmf = os.path.join(kout_dir, f'{sidx}_best_lmscore.ark')
    if not (os.path.isfile(tr) and os.path.isfile(acf)):
        return None
    lines = open(tr).read().strip().split('\n')
    texts = []
    for l in lines:
        toks = l.split(' ')
        del toks[0]
        texts.append(''.join(' ' if t == '<space>' else '' if t == '\n' else t for t in toks))
    acs = [float(l.split()[1]) for l in open(acf).read().strip().split('\n')]
    lms = [float(l.split()[1]) for l in open(lmf).read().strip().split('\n')]
    bi = int(np.argmin(np.array(acs) + 2.0 * np.array(lms)))
    return texts[bi], texts, acs, lms


def load_graph_cached(path, name, cache_path):
    """Load a Graph from pickle cache, building and caching it on first use.

    Parameters
    ----------
    path : str
        Path to the AT&T text FST file (only read on cache miss).
    name : str
        Display name of the graph (``'HCL'`` or ``'G'``).
    cache_path : str
        Pickle cache file; written atomically (``.tmp`` + replace) after
        the graph is built.

    Returns
    -------
    Graph
        The loaded graph, either unpickled from ``cache_path`` or freshly
        parsed from ``path``.
    """
    import pickle
    if os.path.isfile(cache_path):
        print('loading cached graph:', cache_path, flush=True)
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    g = Graph(path, name)
    tmp = cache_path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(g, f, protocol=4)
    os.replace(tmp, cache_path)
    return g


def main():
    """Decode validation kaldiMats with the pure-Python lazy bigram decoder.

    Loads the transition model, symbols and (cached) HCL/G graphs, then for
    each selected session decodes every kaldiMat (optionally limited by
    sentence index via ``--limit``), selects
    ``argmin(ac + nbestSelect*graph)`` from the 128-best list, converts to
    text, compares against the official KaldiOutput selection, accumulates
    CER/WER over validation sentences and prints a summary against the
    official reference numbers.

    Parameters
    ----------
    None
        Options come from the command line: ``--cvPart`` (required),
        ``--limit`` (only sentence indices below it), ``--sessions``
        (comma list), ``--nbestSelect`` (LM weight in selection, default
        2.0).

    Returns
    -------
    None

    Notes
    -----
    Assigns the module-global ``TID2PHONE`` used inside
    ``LazyDecoder.decode`` for phone lookup.
    """
    global TID2PHONE
    ap = argparse.ArgumentParser()
    ap.add_argument('--cvPart', choices=['HeldOutBlocks', 'HeldOutTrials'], required=True)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--sessions', default=None)
    ap.add_argument('--nbestSelect', type=float, default=2.0)
    cli = vars(ap.parse_args())

    tid2pdf, TID2PHONE, tid_lp = load_model(os.path.join(LM_DIR, 'model'))
    chars = load_symbols(os.path.join(LM_DIR, 'chars.txt'))
    chars_by_id = {v: k for k, v in chars.items()}
    hcl = load_graph_cached(os.path.join(CACHE, 'HCL.txt'), 'HCL', os.path.join(CACHE, 'HCL.pkl'))
    g = load_graph_cached(os.path.join(CACHE, 'G.txt'), 'G', os.path.join(CACHE, 'G.pkl'))
    dec = LazyDecoder(hcl, g, tid2pdf, tid_lp)

    sessions = cli['sessions'].split(',') if cli['sessions'] else SESSIONS
    val_err = []
    n_exact, n_tot = 0, 0
    t0 = __import__('time').time()
    n_done = 0
    for sess in sessions:
        mats_dir = os.path.join(ROOT, 'RNNTrainingSteps/Step6_ApplyBigramLM', cli['cvPart'], 'KaldiMatrices', sess)
        if not os.path.isdir(mats_dir):
            continue
        part = scipy.io.loadmat(os.path.join(ROOT, 'RNNTrainingSteps', 'trainTestPartitions_' + cli['cvPart'] + '.mat'))
        val_idx = set(part[sess + '_test'][0].tolist()) if len(part[sess + '_test']) else set()
        sentence_dat = scipy.io.loadmat(os.path.join(ROOT, 'Datasets', sess, 'sentences.mat'))
        mats = sorted([f for f in os.listdir(mats_dir) if f.startswith('kaldiMat_')],
                      key=lambda x: int(x[len('kaldiMat_'):-4]))
        if cli['limit']:
            mats = [m for m in mats if int(m[9:-4]) < cli['limit']]
        for mf in mats:
            sidx = int(mf[len('kaldiMat_'):-4])
            mat = load_kaldi_mat(os.path.join(mats_dir, mf))
            cands = dec.decode(mat)
            n_done += 1
            if not cands:
                print(f'{sess} #{sidx}: 无候选', flush=True)
                continue
            sel = min(cands, key=lambda c: c[0] + cli['nbestSelect'] * c[1])
            text = phones_to_transcript(sel[2], chars_by_id)
            kout = os.path.join(ROOT, 'RNNTrainingSteps/Step6_ApplyBigramLM', cli['cvPart'], 'KaldiOutput', sess)
            ref = read_ref_selection(kout, sidx)
            if ref is not None and sidx in val_idx:
                n_tot += 1
                if text == ref[0]:
                    n_exact += 1
                elif n_tot - n_exact <= 5:
                    print(f'{sess} #{sidx}:\n  mine: {text!r}\n  ref : {ref[0]!r}', flush=True)
            if sidx in val_idx:
                true_text = sentence_dat['sentencePrompt'][sidx, 0][0].replace('>', ' ').replace('~', '.').replace('#', '')
                char_errs = wer(true_text, text)
                word_errs = wer(true_text.split(), text.split())
                val_err.append([char_errs, len(true_text), word_errs, len(true_text.split())])
            if n_done % 20 == 0:
                dt = __import__('time').time() - t0
                print(f'... {n_done} sentences, {dt/n_done:.2f} s/sent', flush=True)
    if val_err:
        concat = np.array(val_err)
        cer = 100 * concat[:, 0].sum() / concat[:, 1].sum()
        werr = 100 * concat[:, 2].sum() / concat[:, 3].sum()
        print(f'\n{cli["cvPart"]}: 自解码 bigram CER={cer:.4f}% WER={werr:.4f}% n={concat.shape[0]}')
        print(f'与包内选句 transcript 完全一致: {n_exact}/{n_tot}')
        print('官方对照: HeldOutBlocks 1.69/6.10, HeldOutTrials 0.80/3.64')


if __name__ == '__main__':
    main()
