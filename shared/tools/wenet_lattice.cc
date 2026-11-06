// wenet_lattice.cc - offline port of the official nejm-brain-to-text word
// decoder: language-model-standalone.py + lm_decoder.cc (DecodeNumpy) +
// WeNet CtcWfstBeamSearch (blank-skip frame loop) + kaldi
// LatticeFasterDecoder, with the torch/redis layer replaced by .npy input.
//   - blank penalty applied exactly like DecodeNumpy: lp[:,0] -= log(bp)
//   - n-best via CompactLattice -> ConvertLattice -> ShortestPath(nbest)
//     -> GetLinearSymbolSequence; lm = -Value1, ac = -Value2 / acs
// The TLG.fst is mmap'd with a sidecar .idx (8 B/state), never loaded.
// usage: wenet_lattice TLG.fst words.txt trials_dir meta.tsv out.tsv
//        [acs=0.325] [bp=90] [beam=17] [lattice_beam=8] [max_active=7000]
//        [min_active=200] [nbest=100] [nbest_prefix=-] [build=0] [skip=1.0]
#include <fst/fst.h>
#include <fst/vector-fst.h>

#include "glog_shim.h"
#include "base/kaldi-common.h"
#include "decoder/lattice-faster-decoder.h"
#include "decoder/lattice-faster-online-decoder.h"
#include "fstext/fstext-lib.h"
#include "itf/decodable-itf.h"
#include "lat/kaldi-lattice.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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
      off.reserve(1024);
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
      fprintf(stderr, "mmap: type=%s ver=%d states=%lld file=%.1f GB\n",
              hdr.FstType().c_str(), hdr.Version(),
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
  inline void arc(int32_t s, int32_t i, fst::StdArc *a) const {
    const uint8_t *p = base + off(s) + state_hdr + (size_t)i * 16;
    memcpy(&a->ilabel, p, 4);
    memcpy(&a->olabel, p + 4, 4);
    float w; memcpy(&w, p + 8, 4); a->weight = w;
    memcpy(&a->nextstate, p + 12, 4);
  }
  inline float final_w(int32_t s) const {
    float w; memcpy(&w, base + off(s), 4); return w;
  }
};

// ---- adapt the mmap'd graph to the OpenFst Fst interface for Kaldi ----
class MappedArcIterator : public fst::ArcIteratorBase<fst::StdArc> {
 public:
  MappedArcIterator(const MappedFst *m, fst::StdArc::StateId s)
      : m_(m), s_(s), i_(0), n_((size_t)m->narcs(s)) {}
  bool Done() const override { return i_ >= n_; }
  const fst::StdArc &Value() const override {
    m_->arc(s_, (int32_t)i_, &a_);
    a_.ilabel = remap_ilabel(a_.ilabel);
    return a_;
  }
  // This TLG (Willett BtoT24 build) labels arcs 1=blank, 2=SIL (which the
  // model emits as class 40 '|'), 3..41 = phones (model classes 1..39).
  // The official decoder reads LogLikelihood(ilabel) = lp[ilabel-1], so
  // remap on the fly: 1->1, 2->41, k->k-1.
  static inline int32_t remap_ilabel(int32_t il) {
    if (il == 1) return 1;
    if (il == 2) return 41;
    if (il >= 3 && il <= 41) return il - 1;
    return 0;
  }
  void Next() override { ++i_; }
  void Reset() override { i_ = 0; }
  void Seek(size_t a) override { i_ = a; }
  size_t Position() const override { return i_; }
  uint8_t Flags() const override { return fst::kArcValueFlags; }
  void SetFlags(uint8, uint8) override {}

 private:
  const MappedFst *m_;
  fst::StdArc::StateId s_;
  size_t i_, n_;
  mutable fst::StdArc a_;
};

class MappedStateIterator : public fst::StateIteratorBase<fst::StdArc> {
 public:
  explicit MappedStateIterator(const MappedFst *m) : m_(m), i_(0) {}
  bool Done() const override {
    return i_ >= (fst::StdArc::StateId)m_->num_states;
  }
  fst::StdArc::StateId Value() const override { return i_; }
  void Next() override { ++i_; }
  void Reset() override { i_ = 0; }

 private:
  const MappedFst *m_;
  fst::StdArc::StateId i_;
};

class MappedFstWrapper : public fst::Fst<fst::StdArc> {
 public:
  explicit MappedFstWrapper(const MappedFst *m) : m_(m) {}
  StateId Start() const override { return m_->start; }
  Weight Final(StateId s) const override {
    float w = m_->final_w(s);
    return (w >= 1e30f) ? Weight::Zero() : Weight(w);
  }
  size_t NumArcs(StateId s) const override { return (size_t)m_->narcs(s); }
  size_t NumInputEpsilons(StateId s) const override {
    size_t c = 0;
    int32_t n = m_->narcs(s);
    fst::StdArc a;
    for (int32_t i = 0; i < n; i++) {
      m_->arc(s, i, &a);
      if (a.ilabel == 0) c++;
    }
    return c;
  }
  size_t NumOutputEpsilons(StateId s) const override {
    size_t c = 0;
    int32_t n = m_->narcs(s);
    fst::StdArc a;
    for (int32_t i = 0; i < n; i++) {
      m_->arc(s, i, &a);
      if (a.olabel == 0) c++;
    }
    return c;
  }
  void InitArcIterator(
      StateId s, fst::ArcIteratorData<fst::StdArc> *data) const override {
    data->base.reset(new MappedArcIterator(m_, s));
  }
  const std::string &Type() const override {
    static const std::string t = "mapped";
    return t;
  }
  MappedFstWrapper *Copy(bool safe = false) const override {
    return new MappedFstWrapper(m_);
  }
  uint64_t Properties(uint64 mask, bool test) const override { return 0; }
  const fst::SymbolTable *InputSymbols() const override { return nullptr; }
  const fst::SymbolTable *OutputSymbols() const override { return nullptr; }
  void InitStateIterator(
      fst::StateIteratorData<fst::StdArc> *data) const override {
    data->base.reset(new MappedStateIterator(m_));
  }

 private:
  const MappedFst *m_;
};

// ---- decodable over one trial's [T,41] log-prob matrix ----
class NpyDecodable : public kaldi::DecodableInterface {
 public:
  NpyDecodable(const float *lp, int32_t T, float scale)
      : lp_(lp), T_(T), scale_(scale) {}
  kaldi::BaseFloat LogLikelihood(int32_t frame, int32_t index) override {
    // index is the fst ilabel, 1-based over the 41 classes (wenet semantics)
    return scale_ * lp_[(size_t)frame * 41 + (index - 1)];
  }
  bool IsLastFrame(int32_t frame) const override { return frame == T_ - 1; }
  int32_t NumFramesReady() const override { return T_; }
  int32_t NumIndices() const override { return 41; }

 private:
  const float *lp_;
  int32_t T_;
  float scale_;
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

static double now_ms() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr,
            "usage: %s TLG.fst words.txt trials_dir meta.tsv out.tsv "
            "[acs=0.325] [bp=90] [beam=17] [lattice_beam=8] "
            "[max_active=7000] [min_active=200] [nbest=100] "
            "[nbest_prefix=-] [build=0] [skip=1.0]\n",
            argv[0]);
    return 1;
  }
  const char *fst_path = argv[1];
  const char *words_path = argv[2];
  const char *trials_dir = argv[3];
  const char *meta_path = argv[4];
  const char *out_path = argv[5];
  const float acs = (argc > 6) ? (float)atof(argv[6]) : 0.325f;
  const float bp = (argc > 7) ? (float)atof(argv[7]) : 90.0f;
  const float beam = (argc > 8) ? (float)atof(argv[8]) : 17.0f;
  const float lattice_beam = (argc > 9) ? (float)atof(argv[9]) : 8.0f;
  const int max_active = (argc > 10) ? atoi(argv[10]) : 7000;
  const int min_active = (argc > 11) ? atoi(argv[11]) : 200;
  const int nbest = (argc > 12) ? atoi(argv[12]) : 100;
  const char *nbest_prefix = (argc > 13) ? argv[13] : "-";
  const bool build = (argc > 14) && atoi(argv[14]);
  const float skip_thresh = (argc > 15) ? (float)atof(argv[15]) : 1.0f;
  const float lnbp = (bp > 1.0f) ? std::log(bp) : 0.0f;

  std::unordered_map<int, std::string> wordmap;
  {
    FILE *f = fopen(words_path, "r");
    if (!f) { perror("words.txt"); return 1; }
    char sym[8192];
    long long id;
    while (fscanf(f, "%8191s %lld", sym, &id) == 2) wordmap[(int)id] = sym;
    fclose(f);
    fprintf(stderr, "words: %zu | acs=%.3f bp=%.1f beam=%.1f lat_beam=%.1f "
                    "max_act=%d min_act=%d nbest=%d skip=%.2f\n",
            wordmap.size(), acs, bp, beam, lattice_beam, max_active,
            min_active, nbest, skip_thresh);
  }

  std::vector<std::pair<std::string, std::string>> trials;
  {
    FILE *f = fopen(meta_path, "r");
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

  MappedFst mf;
  mf.open(fst_path, build);
  MappedFstWrapper wrapper(&mf);

  FILE *out = fopen(out_path, "w");
  if (!out) { perror("out.tsv"); return 1; }
  fprintf(out, "key\tref\thyp\tscore\n");
  FILE *nb = nullptr;
  std::string nbpath;
  if (strcmp(nbest_prefix, "-") != 0 && nbest > 0) {
    nbpath = std::string(nbest_prefix) + ".tsv";
    nb = fopen(nbpath.c_str(), "w");
    if (!nb) { perror("nbest"); return 1; }
  }

  double t_start = now_ms();
  size_t ndec = 0;

  for (size_t ti = 0; ti < trials.size(); ti++) {
    const auto &tr = trials[ti];
    size_t T = 0;
    std::vector<float> logp =
        read_npy41(std::string(trials_dir) + "/" + tr.first + ".npy", T);
    if (T == 0) {
      fprintf(out, "%s\t%s\t\t0.0\n", tr.first.c_str(), tr.second.c_str());
      continue;
    }
    if (lnbp != 0.0f)
      for (size_t t = 0; t < T; t++) logp[t * 41] -= lnbp;

    kaldi::LatticeFasterDecoderConfig dopts;
    dopts.beam = beam;
    dopts.lattice_beam = lattice_beam;
    dopts.max_active = max_active;
    dopts.min_active = min_active;
    kaldi::LatticeFasterOnlineDecoderTpl<fst::Fst<fst::StdArc>> decoder(
        wrapper, dopts);
    decoder.InitDecoding();

    // official CtcWfstBeamSearch::Search frame loop: frames whose blank
    // posterior exceeds skip_thresh are dropped (and re-inserted before a
    // repeated symbol). skip=1.0 (online default) feeds every frame.
    std::vector<float> frames;
    frames.swap(logp);
    if (skip_thresh < 1.0f) {
      frames.clear();
      bool last_blank = false;
      int last_best = 0;
      for (size_t t = 0; t < T; t++) {
        const float *row = logp.data() + t * 41;
        if (std::exp(row[0]) > skip_thresh) {
          last_blank = true;
        } else {
          int cur_best = 0;
          for (int c = 1; c < 41; c++)
            if (row[c] > row[cur_best]) cur_best = c;
          if (cur_best != 0 && last_blank && cur_best == last_best)
            frames.insert(frames.end(), row - 41, row);
          last_best = cur_best;
          frames.insert(frames.end(), row, row + 41);
          last_blank = false;
        }
      }
    }
    int32_t nfr = (int32_t)(frames.size() / 41);
    if (nfr == 0) {
      fprintf(out, "%s\t%s\t\t0.0\n", tr.first.c_str(), tr.second.c_str());
      continue;
    }
    NpyDecodable dcd(frames.data(), nfr, acs);
    decoder.AdvanceDecoding(&dcd, nfr);

    decoder.FinalizeDecoding();
    kaldi::CompactLattice clat;
    decoder.GetLattice(&clat, true);
    kaldi::Lattice lat;
    fst::ConvertLattice(clat, &lat);
    kaldi::Lattice nbest_fst;
    fst::ShortestPath(lat, &nbest_fst, nbest);
    std::vector<kaldi::Lattice> lats;
    fst::ConvertNbestToVector(nbest_fst, &lats);

    std::string best_hyp;
    float best_total = 0.0f;
    int rank = 0;
    for (auto &lf : lats) {
      kaldi::LatticeWeight w;
      std::vector<int32_t> ali, wids;
      fst::GetLinearSymbolSequence(lf, &ali, &wids, &w);
      std::string sent;
      for (int wid : wids) {
        if (wid == 0) continue;
        auto it = wordmap.find(wid);
        if (!sent.empty()) sent += ' ';
        sent += (it != wordmap.end()) ? it->second : "<unk>";
      }
      float lm = -w.Value1();
      float ac = -w.Value2() / acs;
      if (nb)
        fprintf(nb, "%s\t%d\t%.4f\t%.4f\t%s\n", tr.first.c_str(), rank, ac,
                lm, sent.c_str());
      if (rank == 0) {
        best_hyp = sent;
        best_total = acs * ac + lm;
      }
      rank++;
    }
    fprintf(out, "%s\t%s\t%s\t%.3f\n", tr.first.c_str(), tr.second.c_str(),
            best_hyp.c_str(), -best_total);
    ndec++;
    if (ti % 20 == 0) {
      fprintf(stderr, "[%zu/%zu] %s nfr=%d paths=%zu %.1fs\n", ti + 1,
              trials.size(), tr.first.c_str(), nfr, lats.size(),
              (now_ms() - t_start) / 1000.0);
      fflush(out);
      if (nb) fflush(nb);
    }
  }
  fclose(out);
  if (nb) fclose(nb);
  fprintf(stderr, "done: %zu trials in %.1f s\n", ndec,
          (now_ms() - t_start) / 1000.0);
  return 0;
}
