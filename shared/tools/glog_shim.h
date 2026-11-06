// glog_shim.h - the vendored kaldi headers use google-glog syntax
// (google::LogMessage, google::GLOG_WARNING) but the local openfst is built
// without glog, so we satisfy that contract with a null-sink shim.
#ifndef GLOG_SHIM_H_
#define GLOG_SHIM_H_
#include <ostream>
#include <sstream>
namespace google {
enum GlogSeverity { GLOG_INFO, GLOG_WARNING, GLOG_ERROR, GLOG_FATAL };
class LogMessage {
 public:
  LogMessage(const char *file, int line, int sev) {
    (void)file; (void)line; (void)sev;
  }
  std::ostream &stream() { return oss_; }
 private:
  std::ostringstream oss_;
};
}  // namespace google
#endif  // GLOG_SHIM_H_

// kaldi fstext compat: lattice-utils-inl.h references Factor() from kaldi's
// factor.h (not vendored). We never instantiate that conversion direction,
// so a bare declaration is enough to satisfy parsing/linking.
#include <fst/mutable-fst.h>
#include <vector>
namespace fst {
template <class Arc>
void Factor(const Fst<Arc> &ifst, MutableFst<Arc> *fst_factorized,
            std::vector<std::vector<typename Arc::Label>> *symbols);
}
