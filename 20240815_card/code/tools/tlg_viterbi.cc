// tlg_viterbi.cc - frame-synced CTC Viterbi over the composed TLG.fst
// (Willett BtoT24 5-gram, speech_5gram/lang_test, 44 GB / 687M states).
// Memory design: the graph is NEVER loaded into RAM. TLG.fst is mmap'd and a
// sidecar index (TLG.fst.idx, 8-byte offset per state, built by one
// sequential pass) is also mmap'd; both are file-backed and evictable.
// RAM mode (mmap=0) only exists to demonstrate the OOM under the 20 GB cap.
// Input labels: 1=blank, 2=SIL(=model class 40 '|'), 3..41=phones,
// 0=eps free, 42..57=disambiguator free. Class c -> ilabel: 0->1, 1..39->c+2,
// 40->2. Weights and acoustics are both nats, costs simply add.
// usage: tlg_viterbi TLG.fst words.txt trials_dir meta.tsv out.csv
//        [beam=37] [acscale=1.0] [maxtok=100000] [mmap=0] [build=0]
#include <fst/fst.h>
#include <fst/vector-fst.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

struct MappedFst {
  uint8_t *base = nullptr;
  size_t len = 0;
  int32_t start = 0;
  int64_t num_states = 0;
  size_t hdr_len = 0;
  int state_hdr = 12;
  uint64_t *idx = nullptr;

  void build_index(const char *idx_path) {
    for (int hb : {12, 16}) {
      size_t q = hdr_len;
      bool ok = true;
      std::vector<uint64_t> off;
      off.reserve(num_states);
      for (int64_t s = 0; s < num_states; s++) {
        off.push_back((uint64_t)q);
        int32_t narcs;
        memcpy(&narcs, base + q + 4, 4);
        if (narcs < 0 || (size_t)narcs > len) { ok = false; break; }
        q += (size_t)hb + (size_t)narcs * 16;
      }
      if (ok && q == len) {
        state_hdr = hb;
        FILE *f = fopen(idx_path, "wb");
        if (!f) { perror("idx write"); exit(1); }
        if (fwrite(off.data(), 8, off.size(), f) != off.size()) {
          perror("idx fwrite"); exit(1);
        }
        fclose(f);
        fprintf(stderr, "index built: state hdr %d B, idx %.1f GB\n",
                hb, off.size() * 8 / 1e9);
        return;
      }
    }
    fprintf(stderr, "index build: no state layout matched file size\n");
    exit(1);
  }

  void open(const char *path, bool build) {
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) { perror("open TLG"); exit(1); }
    struct stat st;
    if (fstat(fd, &st)) { perror("fstat"); exit(1); }
    len = (size_t)st.st_size;
    base = (uint8_t *)mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { perror("mmap"); exit(1); }
    madvise(base, len, MADV_RANDOM);
    ::close(fd);
    {
      std::ifstream strm(path, std::ios::binary);
      fst::FstHeader hdr;
      if (!hdr.Read(strm, path)) { fprintf(stderr, "hdr.Read FAILED"); exit(1); }
      hdr_len = (size_t)strm.tellg();
      start = (int32_t)hdr.Start();
      num_states = hdr.NumStates();
      fprintf(stderr, "mmap: type=%s arc=%s ver=%d states=%lld file=%.1f GB\n",
              hdr.FstType().c_str(), hdr.ArcType().c_str(), hdr.Version(),
              (long long)num_states, len / 1e9);
      if (hdr.FstType() != "vector") { fprintf(stderr, "not vector fst\n"); exit(1); }
    }
    std::string ipath = std::string(path) + ".idx";
    struct stat ist;
    bool need = (stat(ipath.c_str(), &ist) != 0 ||
                 (size_t)ist.st_size != (size_t)num_states * 8);
    if (need) {
      if (!build) { fprintf(stderr, "index missing (rerun with build=1)\n"); exit(1); }
      build_index(ipath.c_str());
    }
    int ifd = ::open(ipath.c_str(), O_RDONLY);
    if (ifd < 0) { perror("open idx"); exit(1); }
    if (fstat(ifd, &ist)) { perror("fstat idx"); exit(1); }
    idx = (uint64_t *)mmap(nullptr, (size_t)ist.st_size, PROT_READ,
                           MAP_PRIVATE, ifd, 0);
    if (idx == MAP_FAILED) { perror("mmap idx"); exit(1); }
    madvise(idx, (size_t)ist.st_size, MADV_RANDOM);
    ::close(ifd);
    fprintf(stderr, "index ready\n");
  }
  inline uint64_t off(int32_t s) const { return idx[s]; }
  inline int32_t narcs(int32_t s) const {
    int32_t v; memcpy(&v, base + off(s) + 4, 4); return v;
  }
  inline void arc(int32_t s, int32_t i, int32_t *il, int32_t *ol,
                  float *w, int32_t *ns) const {
    const uint8_t *p = base + off(s) + state_hdr + (size_t)i * 16;
    memcpy(il, p, 4); memcpy(ol, p + 4, 4);
    memcpy(w, p + 8, 4); memcpy(ns, p + 12, 4);
  }
  inline float final_w(int32_t s) const {
    float w; memcpy(&w, base + off(s), 4); return w;
  }
};

static std::vector<float> read_npy41(const std::string &path, size_t &rows) {
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) { perror(path.c_str()); exit(1); }
  char magic[6];
  if (fread(magic, 1, 6, f) != 6 || strncmp(magic, "\x93NUMPY", 6)) {
    fprintf(stderr, "not npy: %s\n", path.c_str()); exit(1);
  }
  uint8_t ver[2]; if (fread(ver, 1, 2, f) != 2) exit(1);
  uint8_t hl2[2]; if (fread(hl2, 1, 2, f) != 2) exit(1);
  size_t hl = hl2[0] | ((size_t)hl2[1] << 8);
  std::string header(hl, 0);
  if (fread(&header[0], 1, hl, f) != hl) exit(1);
  const char *sp = strstr(header.c_str(), "'shape':");
  if (!sp) sp = strstr(header.c_str(), "\"shape\":");
  size_t r = 0, c = 0;
  const char *paren = sp ? strchr(sp, '(') : nullptr;
  if (!paren || sscanf(paren + 1, " %zu , %zu", &r, &c) != 2 || c != 41) {
    fprintf(stderr, "bad shape in %s: %s\n", path.c_str(), header.c_str());
    exit(1);
  }
  std::vector<float> data(r * 41);
  if (fread(data.data(), 4, r * 41, f) != r * 41) {
    fprintf(stderr, "short read %s\n", path.c_str()); exit(1);
  }
  fclose(f);
  rows = r;
  return data;
}

struct Tok {
  int32_t state;
  int32_t parent;
  int32_t word;
  float score;   // total cost = graph part + ac part (lower better)
  float ac;      // acoustic part of the cost (-acscale * sum logp)
};

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: %s TLG.fst words.txt trials_dir meta.tsv out.csv "
                    "[beam=37] [acscale=1.0] [maxtok=100000] [mmap=0] [build=0]\n",
            argv[0]);
    return 1;
  }
  const char *fst_path = argv[1];
  const float beam = (argc > 6) ? (float)atof(argv[6]) : 37.0f;
  const float acscale = (argc > 7) ? (float)atof(argv[7]) : 1.0f;
  const size_t maxtok = (argc > 8) ? (size_t)atol(argv[8]) : 100000;
  const bool use_mmap = (argc > 9) && atoi(argv[9]);
  const bool build = (argc > 10) && atoi(argv[10]);
  int nbest = (argc > 11) ? atoi(argv[11]) : 0;   // 0 = 1-best tsv only
  const char *nbest_path = (argc > 12) ? argv[12] : "";
  float blank_pen = (argc > 13) ? (float)atof(argv[13]) : 1.0f;  // log added to blank logp
  const float ln_bp = blank_pen > 1.0f ? logf(blank_pen) : 0.0f;

  std::unordered_map<int64_t, std::string> words;
  {
    FILE *f = fopen(argv[2], "r");
    if (!f) { perror("words.txt"); return 1; }
    char sym[8192]; long long id;
    while (fscanf(f, "%8191s %lld", sym, &id) == 2) words[id] = sym;
    fclose(f);
    fprintf(stderr, "words: %zu\n", words.size());
  }

  int32_t start;
  int64_t num_states;
  auto is_free = [](int32_t l) { return l == 0 || l >= 42; };

  MappedFst mf;
  fst::VectorFst<fst::StdArc> *vf = nullptr;
  if (use_mmap) {
    mf.open(fst_path, build);
    start = mf.start; num_states = mf.num_states;
  } else {
    vf = fst::VectorFst<fst::StdArc>::Read(fst_path);
    if (!vf) { fprintf(stderr, "FST read failed\n"); return 1; }
    start = vf->Start(); num_states = vf->NumStates();
    fprintf(stderr, "ram load: states=%lld\n", (long long)num_states);
  }
  auto NARCS = [&](int32_t s) -> int32_t {
    return use_mmap ? mf.narcs(s) : vf->NumArcs(s);
  };
  auto GETARC = [&](int32_t s, int32_t i, int32_t *il, int32_t *ol,
                    float *w, int32_t *ns) {
    if (use_mmap) {
      mf.arc(s, i, il, ol, w, ns);
    } else {
      fst::ArcIterator<fst::VectorFst<fst::StdArc>> ait(*vf, s);
      ait.Seek(i);
      const auto &a = ait.Value();
      *il = a.ilabel; *ol = a.olabel; *w = (float)a.weight.Value(); *ns = a.nextstate;
    }
  };
  auto FIN = [&](int32_t s) -> float {
    return use_mmap ? mf.final_w(s) : (float)vf->Final(s).Value();
  };

  std::vector<std::pair<std::string, std::string>> trials;
  {
    FILE *f = fopen(argv[4], "r");
    if (!f) { perror("meta.tsv"); return 1; }
    char key[4096], line[16384];
    while (fgets(line, sizeof line, f)) {
      char *tab = strchr(line, '\t');
      if (!tab) continue;
      *tab = 0;
      char *rest = tab + 1;
      rest[strcspn(rest, "\r\n")] = 0;
      trials.emplace_back(line, rest);
    }
    fclose(f);
    fprintf(stderr, "trials: %zu\n", trials.size());
  }

  FILE *out = fopen(argv[5], "w");
  FILE *nb = nbest > 0 ? fopen(nbest_path, "w") : nullptr;
  fprintf(out, "key\tref\thyp\tscore\n");

  std::vector<Tok> pool;
  std::vector<int32_t> frame_start;
  std::unordered_map<int32_t, int32_t> best;

  for (auto &tr : trials) {
    size_t T = 0;
    std::vector<float> logp = read_npy41(std::string(argv[3]) + "/" + tr.first + ".npy", T);
    pool.clear();
    frame_start.clear();
    best.clear();
    pool.push_back({start, -1, 0, 0.0f, 0.0f});
    best[start] = 0;
    frame_start.push_back(0);

    for (size_t t = 0; t < T; t++) {
      for (size_t i = frame_start.back(); i < pool.size(); i++) {
        Tok cur = pool[i];
        int32_t n = NARCS(cur.state);
        for (int32_t a = 0; a < n; a++) {
          int32_t il, ol, ns; float w;
          GETARC(cur.state, a, &il, &ol, &w, &ns);
          if (!(il == 0 || il >= 42)) continue;
          float sc = cur.score + w;
          auto it = best.find(ns);
          if (it == best.end()) {
            best[ns] = (int32_t)pool.size();
            pool.push_back({ns, (int32_t)i, ol, sc, pool[i].ac});
          } else if (sc < pool[it->second].score) {
            pool[it->second] = {ns, (int32_t)i, ol, sc, pool[i].ac};
          }
        }
      }
      std::unordered_map<int32_t, int32_t> next;
      std::vector<Tok> next_pool;
      const float *lp = &logp[t * 41];
      for (size_t i = frame_start.back(); i < pool.size(); i++) {
        int32_t n = NARCS(pool[i].state);
        for (int32_t a = 0; a < n; a++) {
          int32_t il, ol, ns; float w;
          GETARC(pool[i].state, a, &il, &ol, &w, &ns);
          if ((il == 0 || il >= 42) || il < 1 || il > 41) continue;
          int32_t cls = (il == 1) ? 0 : (il == 2) ? 40 : il - 2;
          float acd = (cls == 0) ? -acscale * (lp[cls] + ln_bp)
                                 : -acscale * lp[cls];
          float sc = pool[i].score + w + acd;
          auto it = next.find(ns);
          float nac = pool[i].ac + acd;
          if (it == next.end()) {
            next[ns] = (int32_t)next_pool.size();
            next_pool.push_back({ns, (int32_t)i, ol, sc, nac});
          } else if (sc < next_pool[it->second].score) {
            next_pool[it->second] = {ns, (int32_t)i, ol, sc, nac};
          }
        }
      }
      if (next_pool.empty()) { frame_start.push_back((int32_t)pool.size()); break; }
      float bestsc = INFINITY;
      for (auto &tk : next_pool) bestsc = std::min(bestsc, tk.score);
      std::vector<int32_t> keep;
      for (int32_t i = 0; i < (int32_t)next_pool.size(); i++)
        if (next_pool[i].score < bestsc + beam) keep.push_back(i);
      if (keep.size() > maxtok) {
        std::nth_element(keep.begin(), keep.begin() + maxtok, keep.end(),
                         [&](int32_t a, int32_t b) {
                           return next_pool[a].score < next_pool[b].score;
                         });
        keep.resize(maxtok);
      }
      frame_start.push_back((int32_t)pool.size());
      best.clear();
      for (int32_t i : keep) {
        best[next_pool[i].state] = (int32_t)pool.size();
        pool.push_back(next_pool[i]);
      }
    }

    for (size_t i = frame_start.back(); i < pool.size(); i++) {
      Tok cur = pool[i];
      int32_t n = NARCS(cur.state);
      for (int32_t a = 0; a < n; a++) {
        int32_t il, ol, ns; float w;
        GETARC(cur.state, a, &il, &ol, &w, &ns);
        if (!(il == 0 || il >= 42)) continue;
        float sc = cur.score + w;
        auto it = best.find(ns);
        if (it == best.end()) {
          best[ns] = (int32_t)pool.size();
          pool.push_back({ns, (int32_t)i, ol, sc, pool[i].ac});
        } else if (sc < pool[it->second].score) {
          pool[it->second] = {ns, (int32_t)i, ol, sc, pool[i].ac};
        }
      }
    }
    std::vector<std::pair<float, int32_t>> finals;
    for (size_t i = frame_start.back(); i < pool.size(); i++) {
      float fw = FIN(pool[i].state);
      if (fw >= 1e30f) continue;
      finals.emplace_back(pool[i].score + fw, (int32_t)i);
    }
    std::sort(finals.begin(), finals.end());
    std::string hyp;
    float bestsc = INFINITY;
    if (nb == nullptr) {
      int32_t bi = finals.empty() ? -1 : finals[0].second;
      bestsc = finals.empty() ? INFINITY : finals[0].first;
      std::vector<int32_t> ws;
      for (int32_t i = bi; i >= 0; i = pool[i].parent) {
        if (pool[i].word) ws.push_back(pool[i].word);
      }
      std::reverse(ws.begin(), ws.end());
      for (auto w : ws) {
        if (!hyp.empty()) hyp += ' ';
        hyp += words.count(w) ? words[w] : "<unk>";
      }
    } else {
      std::unordered_map<std::string, int> uniq;
      int tried = 0;
      int emitted = 0;
      for (auto &pr : finals) {
        if (emitted >= nbest) break;
        std::vector<int32_t> ws;
        for (int32_t i = pr.second; i >= 0; i = pool[i].parent) {
          if (pool[i].word) ws.push_back(pool[i].word);
        }
        std::reverse(ws.begin(), ws.end());
        std::string cand;
        for (auto w : ws) {
          if (!cand.empty()) cand += ' ';
          cand += words.count(w) ? words[w] : "<unk>";
        }
        if (++tried >= 50000) break;
        if (uniq.count(cand)) continue;
        uniq[cand] = 1;
        float ac_logp = -pool[pr.second].ac / acscale;
        float graph_logp = -(pr.first - pool[pr.second].ac);
        fprintf(nb, "%s	%d	%.4f	%.4f	%s\n", tr.first.c_str(), emitted,
                ac_logp, graph_logp, cand.c_str());
        if (emitted == 0) { bestsc = pr.first; hyp = cand; }
        emitted++;
      }
    }
    fprintf(out, "%s	%s	%s	%.3f\n", tr.first.c_str(),
            tr.second.c_str(), hyp.c_str(), bestsc);
    if ((&tr - &trials[0]) % 20 == 0)
      fprintf(stderr, "[%zu/%zu] %s len=%zu live=%zu\n",
              &tr - &trials[0] + 1, trials.size(), tr.first.c_str(), T,
              pool.size() - frame_start.back());
  }
  fclose(out);
  fprintf(stderr, "done\n");
  return 0;
}
