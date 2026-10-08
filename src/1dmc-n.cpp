// 1dmc-n.cpp: 1D slab coupled neutron-photon Monte Carlo.
//
// Neutrons : continuous-energy ENDF/B-VIII.0 pointwise data (294 K, resonances included, so self-shielding is
//            automatic). Elastic (isotropic CM, exact kinematics), 39+ discrete inelastic levels per nuclide,
//            continuum inelastic (evaporation), (n,2n), capture and other absorption.
//            Thermal region: bound-hydrogen S(alpha,beta) tables (water, polyethylene) below 5 eV; free-gas
//            target motion (constant-xs approximation) for every other nuclide below 400 kT.
// Photons  : produced by (n,gamma) using measured EGAF cascades, by (n,n') using RIPL-3 level schemes with
//            internal conversion, or supplied as a primary source. Photoelectric (local absorption), Compton
//            (Klein-Nishina, free electron), pair production (annihilation photons). Rayleigh neglected.
// Dose     : ICRP-116 effective dose per fluence (neutron and photon), selectable geometry.
//
// NOT modelled: charged-particle transport / bremsstrahlung (local energy deposition), fission, photoneutrons,
// anisotropic elastic scattering in the CM frame, thermal scattering of non-hydrogen atoms.
//
// Build:  g++ -O3 -std=c++17 -fopenmp src/1dmc-n.cpp -o build/1dmc-n
// Output: JSON on stdout (consumed by gui/1dmc-n_gui.py)

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// Constants and grids
constexpr int NB_P = 600;                  // photon bins (must match tools/build_data.py)
constexpr double EMIN_P = 1e4, EMAX_P = 2e7;
constexpr double EMAX_N = 2e7;
constexpr double E_ANALOG = 1.0;           // eV: below this capture is analog (no implicit capture)
constexpr double E_COLD = 0.5;             // eV: flux-tally boundary "thermal"
constexpr double KT = 0.025335;            // eV at 294 K
constexpr double FG_THRESH = 400.0;        // free-gas treatment below FG_THRESH*kT
constexpr double AVOGADRO = 0.602214076;   // atoms/(barn cm) per (mol/g)
constexpr double ME_EV = 510998.95;
constexpr double MU_MIN = 0.05;            // clamp for the 1/mu fluence estimator at the exit plane
constexpr int NSPEC = 64;                  // exit-spectrum bins
constexpr double SPEC_LO_N = 1e-5, SPEC_HI_N = 2e7, SPEC_LO_P = 1e4, SPEC_HI_P = 2e7;
constexpr int NDOSE = 2000;                // dose-lookup bins (neutron)
constexpr double DOSE_LO = 1e-5, DOSE_HI = 2e7;
constexpr int MAXLEV = 64;

static const double LN_EMIN_P = std::log(EMIN_P), DLN_P = (std::log(EMAX_P) - std::log(EMIN_P)) / NB_P;
inline int pbin(double E) { return std::min(NB_P - 1, std::max(0, (int)((std::log(E) - LN_EMIN_P) / DLN_P))); }
static const double LN_DLO = std::log(DOSE_LO), DLN_D = (std::log(DOSE_HI) - std::log(DOSE_LO)) / NDOSE;
inline int dbin(double E) { return std::min(NDOSE - 1, std::max(0, (int)((std::log(E) - LN_DLO) / DLN_D))); }

// Random numbers: xoshiro256** seeded with splitmix64
struct Rng {
  uint64_t s[4];
  static uint64_t splitmix(uint64_t& x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
  explicit Rng(uint64_t seed) { for (auto& v : s) v = splitmix(seed); }
  static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
  inline uint64_t next() {
    const uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t; s[3] = rotl(s[3], 45);
    return r;
  }
  inline double uni() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
  inline double uopen() { double u; do { u = uni(); } while (u == 0.0); return u; }
};

// Small helpers
static std::string DATA_DIR;
static std::vector<std::string> split(const std::string& s, char d) {
  std::vector<std::string> out; std::string t; std::stringstream ss(s);
  while (std::getline(ss, t, d)) out.push_back(t);
  return out;
}
static std::string lower(std::string s) { for (auto& c : s) c = (char)std::tolower((unsigned char)c); return s; }
static bool file_exists(const std::string& p) { std::ifstream f(p); return f.good(); }

static std::string find_data_dir(const std::string& hint, const char* argv0) {
  std::vector<std::string> c;
  if (!hint.empty()) c.push_back(hint);
  if (const char* e = std::getenv("ONEDMC_N_DATA")) c.push_back(e);
  std::string exe = argv0 ? argv0 : "";
  size_t sl = exe.find_last_of("/\\");
  std::string dir = sl == std::string::npos ? "." : exe.substr(0, sl);
  c.push_back(dir + "/../data"); c.push_back(dir + "/data"); c.push_back("data"); c.push_back("../data");
  for (auto& p : c) if (file_exists(p + "/elements.csv")) return p;
  throw std::runtime_error("data directory not found (use --data DIR or set ONEDMC_N_DATA)");
}

struct V3 { double x, y, z; };
static inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline V3 operator*(double s, V3 a) { return {s * a.x, s * a.y, s * a.z}; }
static inline double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static V3 rot(const V3& u, double mu, double phi) {   // unit vector at polar cosine mu around u
  double s = std::sqrt(std::max(0.0, 1 - mu * mu));
  V3 a = std::fabs(u.x) < 0.9 ? V3{1, 0, 0} : V3{0, 1, 0};
  V3 e1 = cross(u, a); double n = std::sqrt(dot(e1, e1)); e1 = (1.0 / n) * e1;
  V3 e2 = cross(u, e1);
  return mu * u + s * (std::cos(phi) * e1 + std::sin(phi) * e2);
}

// Grid with log-hash for fast index search
struct GridHash {
  const std::vector<double>* g = nullptr;
  std::vector<int> h; double l0 = 0, inv = 0;
  static constexpr int NH = 8192;
  void build(const std::vector<double>& grid) {
    g = &grid; l0 = std::log(grid.front()); double l1 = std::log(grid.back()); inv = NH / (l1 - l0);
    h.assign(NH + 2, 0); size_t j = 0;
    for (int b = 0; b <= NH + 1; ++b) {
      double e = std::exp(l0 + b / inv);
      while (j + 1 < grid.size() && grid[j + 1] <= e) ++j;
      h[b] = (int)j;
    }
  }
  inline int find(double E) const {            // i such that g[i] <= E < g[i+1], clamped to [0, n-2]
    const std::vector<double>& G = *g; int n = (int)G.size();
    if (E <= G[0]) return 0;
    if (E >= G[n - 1]) return n - 2;
    int b = (int)((std::log(E) - l0) * inv); b = std::min(std::max(b, 0), NH);
    int lo = h[b], hi = std::min(h[b + 1] + 1, n - 1);
    while (hi - lo > 1) { int m = (lo + hi) >> 1; if (G[m] <= E) lo = m; else hi = m; }
    return lo;
  }
};

// Data: cascades, nuclides, thermal scattering, photon cross sections
struct Cascade { double p; std::vector<double> E; };     // explicit photon list (eV) with probability

struct CapTrans { int to; double Eg; double cum; };      // EGAF transition, Eg in eV
struct CapCas {
  double unplaced = 0; int capstate = -1;
  std::vector<std::vector<CapTrans>> from;               // indexed by level
};
struct Branch { int fin; double cum; double pgam; };
struct Lvl { double E = 0; std::vector<Branch> br; };    // E in MeV
struct LevelScheme { std::vector<Lvl> lv; };

struct Chan { int type = 0, ripl = -1; double Ex = 0; uint32_t start = 0; std::vector<float> xs; };
enum { C_EL = 0, C_CAP = 1, C_OABS = 2, C_N2N = 3, C_CONT = 4, C_LEV = 5 };

struct XS { double el, cap, oabs, n2n, cont, scat, abs; double lev[MAXLEV]; int nlev; };

struct TSL {
  int nE = 0, nEo = 0, nMu = 0; double emax = 0;
  std::vector<double> E; std::vector<float> sig, Eo, mu;
  std::string name;
  double sigma(double e) const {
    if (e <= E[0]) return sig[0] * std::sqrt(E[0] / std::max(e, 1e-12));
    if (e >= E[nE - 1]) return sig[nE - 1];
    int i = (int)(std::upper_bound(E.begin(), E.end(), e) - E.begin()) - 1;
    double f = (e - E[i]) / (E[i + 1] - E[i]);
    return sig[i] + f * (sig[i + 1] - sig[i]);
  }
};

struct Nuc {
  std::string name; double A = 0, Sn = 0;
  std::vector<double> E; GridHash gh;
  std::vector<float> el, cap, oabs;
  Chan n2n, cont; std::vector<Chan> lev;
  std::vector<double> sc_arr, ab_arr;                    // scatter / absorption on the grid
  std::shared_ptr<CapCas> capcas; std::shared_ptr<LevelScheme> levels;
  std::vector<Cascade> oabs_lines, cap_lines;
};

struct PData { std::string sym; double Aw = 0; std::vector<double> pe, inc, pair; };
struct Element { std::string sym; int Z; double Aw; };
struct Isotope { std::string sym; int A; double abund, mass, Sn; };

struct DoseTable { std::vector<double> E, v; };
static double dose_interp(const DoseTable& t, double E_MeV) {
  if (E_MeV <= t.E.front()) return t.v.front();
  if (E_MeV >= t.E.back()) return t.v.back();
  size_t i = std::upper_bound(t.E.begin(), t.E.end(), E_MeV) - t.E.begin() - 1;
  double f = std::log(E_MeV / t.E[i]) / std::log(t.E[i + 1] / t.E[i]);
  return std::exp(std::log(t.v[i]) + f * (std::log(t.v[i + 1]) - std::log(t.v[i])));
}
static DoseTable load_dose(const std::string& file, const std::string& geom) {
  std::ifstream f(DATA_DIR + "/dose/" + file);
  if (!f) throw std::runtime_error("cannot open dose table " + file);
  std::string line; int col = -1; DoseTable t;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto p = split(line, ',');
    if (col < 0) {
      for (size_t i = 0; i < p.size(); ++i) if (p[i] == geom) col = (int)i;
      if (col < 0) throw std::runtime_error("unknown dose geometry '" + geom + "' (AP PA LLAT RLAT ROT ISO)");
      continue;
    }
    t.E.push_back(std::atof(p[0].c_str())); t.v.push_back(std::atof(p[col].c_str()));
  }
  return t;
}

static std::vector<Element> ELEMENTS;
static std::vector<Isotope> ISOTOPES;
static std::map<std::string, std::vector<Cascade>> OABS_LINES, CAPLINE_FALLBACK;
static bool USE_DETAILED_GAMMA = true;

static void load_tables() {
  std::string line;
  { std::ifstream f(DATA_DIR + "/elements.csv"); std::getline(f, line);
    while (std::getline(f, line)) { auto p = split(line, ','); if (p.size() >= 3) ELEMENTS.push_back({p[0], std::atoi(p[1].c_str()), std::atof(p[2].c_str())}); } }
  { std::ifstream f(DATA_DIR + "/isotopes.csv"); std::getline(f, line);
    while (std::getline(f, line)) { auto p = split(line, ','); if (p.size() >= 5) ISOTOPES.push_back({p[0], std::atoi(p[1].c_str()), std::atof(p[2].c_str()), std::atof(p[3].c_str()), std::atof(p[4].c_str())}); } }
  std::ifstream g(DATA_DIR + "/gamma/capture_lines.csv");
  if (g) {
    std::getline(g, line);
    while (std::getline(g, line)) {
      if (line.empty() || line[0] == '#') continue;
      auto p = split(line, ',');
      if (p.size() < 4) continue;
      Cascade c; c.p = std::atof(p[2].c_str());
      for (auto& e : split(p[3], ';')) c.E.push_back(std::atof(e.c_str()) * 1e6);
      (p[1] == "oabs" ? OABS_LINES : CAPLINE_FALLBACK)[p[0]].push_back(c);
    }
  }
}
static int find_element(const std::string& s) { for (size_t i = 0; i < ELEMENTS.size(); ++i) if (ELEMENTS[i].sym == s) return (int)i; return -1; }
static int find_isotope(const std::string& s) { for (size_t i = 0; i < ISOTOPES.size(); ++i) if (ISOTOPES[i].sym + std::to_string(ISOTOPES[i].A) == s) return (int)i; return -1; }

static std::shared_ptr<CapCas> load_capcas(const std::string& name) {
  std::ifstream f(DATA_DIR + "/gamma/capture/" + name + ".csv");
  if (!f) return nullptr;
  auto c = std::make_shared<CapCas>();
  std::string line; std::vector<std::array<double, 4>> T; std::map<int, double> Elev; int maxidx = 0;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto p = split(line, ',');
    if (p[0] == "S") c->capstate = std::atoi(p[2].c_str());
    else if (p[0] == "L") { int i = std::atoi(p[1].c_str()); Elev[i] = std::atof(p[2].c_str()); maxidx = std::max(maxidx, i); }
    else if (p[0] == "T") { T.push_back({(double)std::atoi(p[1].c_str()), (double)std::atoi(p[2].c_str()), std::atof(p[3].c_str()) * 1e3, std::atof(p[4].c_str())}); }
  }
  if (c->capstate < 0 || T.empty()) return nullptr;
  c->from.assign(maxidx + 2, {});
  std::vector<double> tot(maxidx + 2, 0.0);
  for (auto& t : T) { int a = (int)t[0]; if (a < 0 || a > maxidx) continue; tot[a] += t[3]; }
  for (auto& t : T) {
    int a = (int)t[0]; if (a < 0 || a > maxidx) continue;
    auto& v = c->from[a]; double prev = v.empty() ? 0.0 : v.back().cum;
    v.push_back({(int)t[1], t[2], prev + t[3] / tot[a]});
  }
  double pprim = c->capstate <= maxidx ? tot[c->capstate] : 0.0;
  c->unplaced = std::max(0.0, 1.0 - std::min(1.0, pprim));
  return c;
}
static std::shared_ptr<LevelScheme> load_levels(const std::string& name) {
  std::ifstream f(DATA_DIR + "/gamma/levels/" + name + ".csv");
  if (!f) return nullptr;
  auto ls = std::make_shared<LevelScheme>();
  std::string line; std::map<int, std::vector<std::array<double, 3>>> br; std::map<int, double> E; int maxidx = 0;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto p = split(line, ',');
    if (p[0] == "L") { int i = std::atoi(p[1].c_str()); E[i] = std::atof(p[2].c_str()); maxidx = std::max(maxidx, i); }
    else if (p[0] == "B") br[std::atoi(p[1].c_str())].push_back({(double)std::atoi(p[2].c_str()), std::atof(p[3].c_str()), std::atof(p[4].c_str())});
  }
  ls->lv.assign(maxidx + 1, Lvl());
  for (auto& kv : E) ls->lv[kv.first].E = kv.second;
  for (auto& kv : br) {
    double s = 0; for (auto& b : kv.second) s += b[1];
    if (s <= 0) continue;
    double run = 0;
    for (auto& b : kv.second) { run += b[1] / s; ls->lv[kv.first].br.push_back({(int)b[0], run, 1.0 / (1.0 + b[2])}); }
  }
  return ls;
}

template <class T> static void rd(std::ifstream& f, T& v) { f.read(reinterpret_cast<char*>(&v), sizeof(T)); }

static void iso_xs(const Nuc& n, const TSL* tsl, double E, XS& x);

static std::unique_ptr<Nuc> load_nuc(const Isotope& iso) {
  auto n = std::make_unique<Nuc>(); n->name = iso.sym + std::to_string(iso.A);
  std::ifstream f(DATA_DIR + "/neutron/" + n->name + ".nbin", std::ios::binary);
  if (!f) throw std::runtime_error("missing neutron data file for " + n->name + " (run tools/build_neutron.py)");
  char magic[8]; f.read(magic, 8);
  if (std::string(magic, 8) != "MCSHNP01") throw std::runtime_error("bad neutron data file " + n->name);
  uint32_t nE, nCh; double A, Sn; rd(f, nE); rd(f, nCh); rd(f, A); rd(f, Sn);
  n->A = A; n->Sn = Sn; n->E.resize(nE); f.read(reinterpret_cast<char*>(n->E.data()), nE * 8);
  n->el.assign(nE, 0); n->cap.assign(nE, 0); n->oabs.assign(nE, 0);
  for (uint32_t c = 0; c < nCh; ++c) {
    int32_t type, ripl; double ex; uint32_t start, count;
    rd(f, type); rd(f, ripl); rd(f, ex); rd(f, start); rd(f, count);
    std::vector<float> xs(count); f.read(reinterpret_cast<char*>(xs.data()), count * 4);
    if (type == C_EL || type == C_CAP || type == C_OABS) {
      auto& dst = type == C_EL ? n->el : (type == C_CAP ? n->cap : n->oabs);
      for (uint32_t k = 0; k < count; ++k) dst[start + k] = xs[k];
    } else {
      Chan ch; ch.type = type; ch.ripl = ripl; ch.Ex = ex; ch.start = start; ch.xs = std::move(xs);
      if (type == C_N2N) n->n2n = ch; else if (type == C_CONT) n->cont = ch; else n->lev.push_back(ch);
    }
  }
  if ((int)n->lev.size() > MAXLEV) throw std::runtime_error("too many levels in " + n->name);
  n->gh.build(n->E);
  n->capcas = USE_DETAILED_GAMMA ? load_capcas(n->name) : nullptr;
  n->levels = USE_DETAILED_GAMMA ? load_levels(n->name) : nullptr;
  if (OABS_LINES.count(n->name)) n->oabs_lines = OABS_LINES[n->name];
  if (CAPLINE_FALLBACK.count(n->name)) n->cap_lines = CAPLINE_FALLBACK[n->name];
  // scatter / absorption on the isotope grid (free-atom data; thermal scattering is applied per material)
  n->sc_arr.resize(nE); n->ab_arr.resize(nE);
  XS x;
  for (uint32_t j = 0; j < nE; ++j) {
    // evaluate exactly at grid point j
    double e = n->E[j]; iso_xs(*n, nullptr, e, x);
    n->sc_arr[j] = x.scat; n->ab_arr[j] = x.abs;
  }
  return n;
}

static std::unique_ptr<TSL> load_tsl(const std::string& name) {
  std::ifstream f(DATA_DIR + "/thermal/" + name + ".tsl", std::ios::binary);
  if (!f) throw std::runtime_error("missing thermal scattering table '" + name + "'");
  char magic[8]; f.read(magic, 8);
  if (std::string(magic, 8) != "MCSHTSL1") throw std::runtime_error("bad thermal table " + name);
  auto t = std::make_unique<TSL>(); t->name = name; uint32_t a, b, c; double awr;
  rd(f, a); rd(f, b); rd(f, c); rd(f, t->emax); rd(f, awr);
  t->nE = a; t->nEo = b; t->nMu = c;
  t->E.resize(a); f.read(reinterpret_cast<char*>(t->E.data()), a * 8);
  t->sig.resize(a); f.read(reinterpret_cast<char*>(t->sig.data()), a * 4);
  t->Eo.resize((size_t)a * b); f.read(reinterpret_cast<char*>(t->Eo.data()), (size_t)a * b * 4);
  t->mu.resize((size_t)a * b * c); f.read(reinterpret_cast<char*>(t->mu.data()), (size_t)a * b * c * 4);
  return t;
}
static PData load_photon(const Element& e) {
  PData d; d.sym = e.sym; d.Aw = e.Aw;
  std::ifstream f(DATA_DIR + "/photon/" + e.sym + ".csv");
  if (!f) throw std::runtime_error("missing photon data file for " + e.sym);
  std::string line; bool header = false;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (!header) { header = true; continue; }
    auto p = split(line, ',');
    d.pe.push_back(std::atof(p[1].c_str())); d.inc.push_back(std::atof(p[2].c_str())); d.pair.push_back(std::atof(p[4].c_str()));
  }
  if ((int)d.pe.size() != NB_P) throw std::runtime_error("unexpected photon grid in " + e.sym);
  return d;
}

// Cross-section evaluation for one isotope
static inline double chv(const Chan& c, int i, double f) {
  const int n = (int)c.xs.size(); const int s = (int)c.start;
  if (i + 1 < s || n == 0) return 0.0;
  double a = i >= s ? c.xs[std::min(i - s, n - 1)] : 0.0;
  double b = (i + 1 - s) < n ? c.xs[i + 1 - s] : c.xs[n - 1];
  return a + f * (b - a);
}
static inline double lin(const std::vector<float>& v, int i, double f) { return v[i] + f * ((double)v[i + 1] - v[i]); }

static void iso_xs(const Nuc& n, const TSL* tsl, double E, XS& x) {
  int i = n.gh.find(E);
  double f = (E - n.E[i]) / (n.E[i + 1] - n.E[i]); f = std::min(1.0, std::max(0.0, f));
  x.el = lin(n.el, i, f); x.cap = lin(n.cap, i, f); x.oabs = lin(n.oabs, i, f);
  x.n2n = chv(n.n2n, i, f); x.cont = chv(n.cont, i, f);
  if (tsl && E < tsl->emax) x.el = tsl->sigma(E);
  double sl = 0; x.nlev = (int)n.lev.size();
  for (int k = 0; k < x.nlev; ++k) { x.lev[k] = chv(n.lev[k], i, f); sl += x.lev[k]; }
  x.scat = x.el + sl + x.n2n + x.cont; x.abs = x.cap + x.oabs;
}

// Fast scatter / absorption macroscopic-weight lookup (no channel detail): used to pick the target isotope.
static inline void iso_sc_ab(const Nuc& n, const TSL* tsl, double E, double& sc, double& ab) {
  int i = n.gh.find(E);
  double f = (E - n.E[i]) / (n.E[i + 1] - n.E[i]); f = std::min(1.0, std::max(0.0, f));
  sc = n.sc_arr[i] + f * (n.sc_arr[i + 1] - n.sc_arr[i]);
  ab = n.ab_arr[i] + f * (n.ab_arr[i + 1] - n.ab_arr[i]);
  if (tsl && E < tsl->emax) sc += tsl->sigma(E) - lin(n.el, i, f);
}

// Materials
struct MatDef { std::string name; double rho; std::vector<std::pair<std::string, double>> mass_frac; };

static std::vector<MatDef> builtin_materials() {
  return {
      {"water", 1.0, {{"H", 0.111894}, {"O", 0.888106}}},
      {"polyethylene", 0.94, {{"H", 0.143711}, {"C", 0.856289}}},
      {"paraffin", 0.93, {{"H", 0.14862}, {"C", 0.85138}}},
      {"borated_pe_5", 1.0, {{"H", 0.136525}, {"C", 0.813475}, {"B", 0.05}}},
      {"boron_carbide", 2.52, {{"B", 0.7826}, {"C", 0.2174}}},
      {"graphite", 1.7, {{"C", 1.0}}},
      {"concrete", 2.3, {{"H", 0.0221}, {"C", 0.002484}, {"O", 0.57493}, {"Na", 0.015208}, {"Mg", 0.001266},
                         {"Al", 0.019953}, {"Si", 0.304627}, {"K", 0.010045}, {"Ca", 0.042951}, {"Fe", 0.006435}}},
      {"iron", 7.874, {{"Fe", 1.0}}},
      {"stainless_steel_304", 8.0, {{"Fe", 0.695}, {"Cr", 0.19}, {"Ni", 0.095}, {"Mn", 0.02}}},
      {"lead", 11.34, {{"Pb", 1.0}}},
      {"tungsten", 19.3, {{"W", 1.0}}},
      {"copper", 8.96, {{"Cu", 1.0}}},
      {"aluminum", 2.699, {{"Al", 1.0}}},
      {"cadmium", 8.65, {{"Cd", 1.0}}},
  };
}
// default thermal scattering assignments: material -> {isotope, table}
static std::map<std::string, std::vector<std::pair<std::string, std::string>>> default_tsl() {
  return {{"water", {{"H1", "hh2o"}}}, {"polyethylene", {{"H1", "hpoly"}}}, {"paraffin", {{"H1", "hpoly"}}},
          {"borated_pe_5", {{"H1", "hpoly"}}}};
}

struct Material {
  std::string name; double rho = 0;
  std::vector<int> nuc;                       // indices into NUC
  std::vector<double> dens;                   // atoms/(barn cm)
  std::vector<int> tsl;                       // per isotope: index into TSLS or -1
  std::vector<double> U, ust, usab;           // union energy grid; total and absorption macroscopic xs (1/cm)
  GridHash uh;
  std::vector<double> g_cum, gst;             // photon: [bin*3 + {pe,inc,pair}] cumulative; total
  std::vector<std::string> notes;
};

static std::vector<std::unique_ptr<Nuc>> NUC;
static std::map<std::string, int> NUC_IDX;
static std::vector<std::unique_ptr<TSL>> TSLS;
static std::map<std::string, int> TSL_IDX;
static std::vector<PData> PH;
static std::map<std::string, int> PH_IDX;

static int get_nuc(int iso_idx) {
  const auto& iso = ISOTOPES[iso_idx]; std::string n = iso.sym + std::to_string(iso.A);
  auto it = NUC_IDX.find(n); if (it != NUC_IDX.end()) return it->second;
  NUC.push_back(load_nuc(iso)); NUC_IDX[n] = (int)NUC.size() - 1; return (int)NUC.size() - 1;
}
static int get_tsl(const std::string& name) {
  auto it = TSL_IDX.find(name); if (it != TSL_IDX.end()) return it->second;
  TSLS.push_back(load_tsl(name)); TSL_IDX[name] = (int)TSLS.size() - 1; return (int)TSLS.size() - 1;
}
static int get_ph(int el_idx) {
  auto it = PH_IDX.find(ELEMENTS[el_idx].sym); if (it != PH_IDX.end()) return it->second;
  PH.push_back(load_photon(ELEMENTS[el_idx])); PH_IDX[ELEMENTS[el_idx].sym] = (int)PH.size() - 1; return (int)PH.size() - 1;
}

static Material build_material(const MatDef& d, const std::vector<std::pair<std::string, std::string>>& tsl_req) {
  Material m; m.name = d.name; m.rho = d.rho;
  double sum = 0; for (auto& p : d.mass_frac) sum += p.second;
  if (sum <= 0 || d.rho <= 0) throw std::runtime_error("material '" + d.name + "' needs positive density and mass fractions");
  std::map<int, double> ndens, edens;
  for (auto& p : d.mass_frac) {
    double w = p.second / sum;
    int ei = find_element(p.first);
    if (ei >= 0) {
      double Ne = w * d.rho * AVOGADRO / ELEMENTS[ei].Aw;
      edens[ei] += Ne;
      for (size_t i = 0; i < ISOTOPES.size(); ++i) if (ISOTOPES[i].sym == p.first) ndens[(int)i] += Ne * ISOTOPES[i].abund;
    } else {
      int ii = find_isotope(p.first);
      if (ii < 0) throw std::runtime_error("unknown element/isotope '" + p.first + "' in material '" + d.name + "'");
      double Ni = w * d.rho * AVOGADRO / ISOTOPES[ii].mass;
      ndens[ii] += Ni; edens[find_element(ISOTOPES[ii].sym)] += Ni;
    }
  }
  if (ndens.size() > 60) throw std::runtime_error("material '" + d.name + "' has too many isotopes (max 60)");
  for (auto& kv : ndens) {
    m.nuc.push_back(get_nuc(kv.first)); m.dens.push_back(kv.second); m.tsl.push_back(-1);
    for (auto& tr : tsl_req)
      if (tr.first == ISOTOPES[kv.first].sym + std::to_string(ISOTOPES[kv.first].A)) {
        m.tsl.back() = get_tsl(tr.second);
        m.notes.push_back(tr.first + " in " + d.name + ": S(a,b) table '" + tr.second + "' below " +
                          std::to_string(TSLS[m.tsl.back()]->emax).substr(0, 4) + " eV");
      }
  }
  // union energy grid for macroscopic total / absorption cross sections
  std::vector<double> U;
  for (size_t k = 0; k < m.nuc.size(); ++k) {
    const Nuc& n = *NUC[m.nuc[k]]; U.insert(U.end(), n.E.begin(), n.E.end());
    if (m.tsl[k] >= 0) { const TSL& t = *TSLS[m.tsl[k]]; for (double e : t.E) if (e <= t.emax) U.push_back(e); U.push_back(t.emax); }
  }
  std::sort(U.begin(), U.end()); U.erase(std::unique(U.begin(), U.end()), U.end());
  m.U = U; m.ust.assign(U.size(), 0.0); m.usab.assign(U.size(), 0.0);
  std::vector<int> ip(m.nuc.size(), 0);
  for (size_t u = 0; u < U.size(); ++u) {
    double Eu = U[u], sc = 0, ab = 0;
    for (size_t k = 0; k < m.nuc.size(); ++k) {
      const Nuc& n = *NUC[m.nuc[k]]; int& i = ip[k]; const int ng = (int)n.E.size();
      while (i + 1 < ng - 1 && n.E[i + 1] <= Eu) ++i;
      double f = (Eu - n.E[i]) / (n.E[i + 1] - n.E[i]); f = std::min(1.0, std::max(0.0, f));
      double s = n.sc_arr[i] + f * (n.sc_arr[i + 1] - n.sc_arr[i]);
      double a = n.ab_arr[i] + f * (n.ab_arr[i + 1] - n.ab_arr[i]);
      if (m.tsl[k] >= 0 && Eu < TSLS[m.tsl[k]]->emax) s += TSLS[m.tsl[k]]->sigma(Eu) - lin(n.el, i, f);
      sc += m.dens[k] * s; ab += m.dens[k] * a;
    }
    m.ust[u] = sc + ab; m.usab[u] = ab;
  }
  m.uh.build(m.U);
  // photon macroscopic cross sections
  m.g_cum.resize(NB_P * 3); m.gst.resize(NB_P);
  for (int b = 0; b < NB_P; ++b) {
    double pe = 0, inc = 0, pr = 0;
    for (auto& kv : edens) { const PData& p = PH[get_ph(kv.first)]; pe += kv.second * p.pe[b]; inc += kv.second * p.inc[b]; pr += kv.second * p.pair[b]; }
    m.g_cum[b * 3] = pe; m.g_cum[b * 3 + 1] = pe + inc; m.g_cum[b * 3 + 2] = pe + inc + pr; m.gst[b] = pe + inc + pr;
  }
  return m;
}
// NB: Material holds a GridHash pointing at its own U vector; Materials are stored in a std::vector that is
// fully reserved before use, and the hash is rebuilt after the final move (see main).

static inline void mlook(const Material& M, double E, double& st, double& sab) {
  int i = M.uh.find(E);
  double f = (E - M.U[i]) / (M.U[i + 1] - M.U[i]); f = std::min(1.0, std::max(0.0, f));
  st = M.ust[i] + f * (M.ust[i + 1] - M.ust[i]);
  sab = M.usab[i] + f * (M.usab[i + 1] - M.usab[i]);
}

// Problem and tallies
struct Region { double x0, x1; int mat; double imp; };
enum class Src { NMono, Cf252, U235, NThermal, GMono, Cs137, Co60 };
inline bool src_is_photon(Src s) { return s == Src::GMono || s == Src::Cs137 || s == Src::Co60; }

struct Problem {
  std::vector<Material> mats; std::vector<Region> regions;
  std::vector<std::pair<std::string, double>> layers; std::vector<int> layer_mat;
  double total = 0;
  Src src = Src::NMono; double E0_MeV = 2.0; bool cosine = false, analog = false, no_gamma = false;
  double split = 1.0; int nb = 50; double dx = 0;
  double w_low = 1e-2, w_surv = 2e-2;
  std::vector<double> dose_n, dose_g;
};

struct Tally {
  double nsrc = 0;
  double n_trans = 0, n_refl = 0, n_abs = 0, n_unc = 0, n2n_extra = 0;
  double g_trans = 0, g_refl = 0, g_abs = 0, g_unc = 0, pair_extra = 0;
  double dose_inc = 0, dose_n = 0, dose_g = 0, unc_analytic = 0;
  double prod_cap = 0, prod_inel = 0, prod_cap_E = 0, prod_inel_E = 0;
  double cap_events = 0, cap_detailed = 0;
  std::vector<double> flux_f, flux_i, flux_t, flux_g, absprof, sp_nt, sp_nr, sp_gt, sp_gr, sp_gp;
  explicit Tally(int nb) : flux_f(nb, 0.0), flux_i(nb, 0.0), flux_t(nb, 0.0), flux_g(nb, 0.0), absprof(nb, 0.0),
                           sp_nt(NSPEC, 0.0), sp_nr(NSPEC, 0.0), sp_gt(NSPEC, 0.0), sp_gr(NSPEC, 0.0), sp_gp(NSPEC, 0.0) {}
};

struct Particle { double x, mu, E, w; int region; bool photon, collided, secondary; };

static int spec_bin(double E, double lo, double hi) {
  double f = (std::log10(E) - std::log10(lo)) / (std::log10(hi) - std::log10(lo));
  return std::min(NSPEC - 1, std::max(0, (int)(f * NSPEC)));
}

// Physics helpers
static double sample_maxwell(Rng& r, double T) {
  double r1 = r.uopen(), r2 = r.uopen(), r3 = r.uni(), c = std::cos(M_PI / 2 * r3);
  return -T * (std::log(r1) + std::log(r2) * c * c);
}
static double sample_watt(Rng& r, double a, double b) {
  double w = sample_maxwell(r, a);
  return w + 0.25 * a * a * b + (2 * r.uni() - 1) * std::sqrt(a * a * b * w);
}
static void two_body(Rng& rng, double E, double A, double Ex_MeV, double& Eout, double& mul) {
  double Ecm = E * A / (A + 1) - Ex_MeV * 1e6;
  if (Ecm < 0) Ecm = 0;
  double En = Ecm * A / (A + 1), Ev = E / ((A + 1) * (A + 1)), muc = 2 * rng.uni() - 1;
  Eout = En + Ev + 2 * muc * std::sqrt(En * Ev);
  if (Eout < 1e-12) { Eout = 1e-12; mul = 1.0; return; }
  mul = std::max(-1.0, std::min(1.0, (std::sqrt(Ev) + std::sqrt(En) * muc) / std::sqrt(Eout)));
}
static inline double rotate_mu(Rng& rng, double mu, double mul) {
  double s1 = std::sqrt(std::max(0.0, 1 - mu * mu)), s2 = std::sqrt(std::max(0.0, 1 - mul * mul));
  return std::max(-1.0, std::min(1.0, mu * mul + s1 * s2 * std::cos(2 * M_PI * rng.uni())));
}
static double sample_evap(Rng& rng, double Emax, double T) {
  double fmax = Emax > T ? T * std::exp(-1.0) : Emax * std::exp(-Emax / T), x;
  do { x = Emax * rng.uni(); } while (rng.uni() * fmax > x * std::exp(-x / T));
  return x;
}

// Elastic scattering off a thermally moving target (free gas, constant-xs approximation).
static void elastic_free_gas(Rng& rng, double& E, double& mu, double A) {
  V3 u{mu, std::sqrt(std::max(0.0, 1 - mu * mu)), 0.0};
  V3 Vn = std::sqrt(E) * u, Vt{0, 0, 0};
  double beta_vn = std::sqrt(A * E / KT), alpha = 1.0 / (1.0 + std::sqrt(M_PI) * beta_vn / 2.0), bvt2, mut;
  for (;;) {
    double r1 = rng.uopen(), r2 = rng.uopen();
    if (rng.uni() < alpha) bvt2 = -std::log(r1 * r2);
    else { double c = std::cos(M_PI / 2 * rng.uni()); bvt2 = -std::log(r1) - std::log(r2) * c * c; }
    double bvt = std::sqrt(bvt2); mut = 2 * rng.uni() - 1;
    double acc = std::sqrt(std::max(0.0, beta_vn * beta_vn + bvt2 - 2 * beta_vn * bvt * mut)) / (beta_vn + bvt);
    if (rng.uni() < acc) break;
  }
  Vt = std::sqrt(bvt2 * KT / A) * rot(u, mut, 2 * M_PI * rng.uni());
  V3 Vcm = (1.0 / (A + 1)) * (Vn + A * Vt), Vr = Vn - Vcm;
  double sp = std::sqrt(dot(Vr, Vr));
  V3 nd = rot((1.0 / sp) * Vr, 2 * rng.uni() - 1, 2 * M_PI * rng.uni());
  V3 V2 = sp * nd + Vcm; double v2 = dot(V2, V2);
  E = std::max(v2, 1e-12); mu = std::max(-1.0, std::min(1.0, V2.x / std::sqrt(std::max(v2, 1e-30))));
}

// Thermal scattering from tabulated S(alpha,beta) (equiprobable-cosine, skewed outgoing-energy tables)
static void tsl_scatter(const TSL& t, Rng& rng, double E, double& Eout, double& mul) {
  int i = (int)(std::upper_bound(t.E.begin(), t.E.end(), E) - t.E.begin()) - 1;
  i = std::min(std::max(i, 0), t.nE - 2);
  double f = (E - t.E[i]) / (t.E[i + 1] - t.E[i]); f = std::min(1.0, std::max(0.0, f));
  const int n = t.nEo; double r = rng.uni() * (n - 3); int j;
  if (r < 0.1) j = 0; else if (r < 0.5) j = 1;
  else if (r > n - 3 - 0.1) j = n - 1; else if (r > n - 3 - 0.5) j = n - 2;
  else j = (int)(r - 0.5) + 2;
  j = std::min(std::max(j, 0), n - 1);
  Eout = (1 - f) * t.Eo[(size_t)i * n + j] + f * t.Eo[(size_t)(i + 1) * n + j];
  Eout = std::max(Eout, 1e-8);
  int k = std::min((int)(rng.uni() * t.nMu), t.nMu - 1);
  double m0 = t.mu[((size_t)i * n + j) * t.nMu + k], m1 = t.mu[((size_t)(i + 1) * n + j) * t.nMu + k];
  mul = std::max(-1.0, std::min(1.0, (1 - f) * m0 + f * m1));
}

static void emit_photons(Tally& T, std::vector<Particle>& bank, const Particle& q, const std::vector<double>& E_list,
                         bool from_cap, Rng& rng) {
  for (double e : E_list) {
    if (e < EMIN_P) continue;
    e = std::min(e, EMAX_P * 0.999);
    bank.push_back(Particle{q.x, 2 * rng.uni() - 1, e, q.w, q.region, true, false, true});
    (from_cap ? T.prod_cap : T.prod_inel) += q.w;
    (from_cap ? T.prod_cap_E : T.prod_inel_E) += q.w * e * 1e-6;
    T.sp_gp[spec_bin(e, SPEC_LO_P, SPEC_HI_P)] += q.w;
  }
}

static void generic_capture_cascade(Tally& T, std::vector<Particle>& bank, const Particle& q, const Nuc& n, Rng& rng) {
  const double Sn = n.Sn * 1e6; int ng = Sn >= 5.5e6 ? 3 : 2;
  double E1 = Sn * (0.5 + 0.5 * std::pow(rng.uni(), 0.7)), rest = Sn - E1;
  std::vector<double> list{E1};
  if (ng == 2) list.push_back(rest);
  else { double c = rng.uni(); list.push_back(rest * c); list.push_back(rest * (1 - c)); }
  emit_photons(T, bank, q, list, true, rng);
}
static void capture_photons(Tally& T, std::vector<Particle>& bank, const Particle& q, const Nuc& n, int type, Rng& rng) {
  if (type == C_OABS) {
    double u = rng.uni(), acc = 0;
    for (auto& c : n.oabs_lines) { acc += c.p; if (u < acc) { emit_photons(T, bank, q, c.E, true, rng); return; } }
    return;
  }
  T.cap_events += q.w;
  if (n.capcas && rng.uni() >= n.capcas->unplaced) {      // measured EGAF cascade
    T.cap_detailed += q.w;
    const CapCas& c = *n.capcas; std::vector<double> evs; int cur = c.capstate;
    for (int guard = 0; guard < 60 && cur >= 0 && cur < (int)c.from.size() && !c.from[cur].empty(); ++guard) {
      const auto& v = c.from[cur]; double u = rng.uni(); size_t j = 0;
      while (j + 1 < v.size() && u >= v[j].cum) ++j;
      evs.push_back(v[j].Eg); cur = v[j].to;
    }
    emit_photons(T, bank, q, evs, true, rng);
    return;
  }
  if (!n.cap_lines.empty()) {                              // hand-entered lines (data/gamma/capture_lines.csv)
    double u = rng.uni(), acc = 0;
    for (auto& c : n.cap_lines) { acc += c.p; if (u < acc) { emit_photons(T, bank, q, c.E, true, rng); return; } }
  }
  generic_capture_cascade(T, bank, q, n, rng);
}
static void level_cascade(Tally& T, std::vector<Particle>& bank, const Particle& q, const Nuc& n, const Chan& ch, Rng& rng) {
  std::vector<double> evs;
  if (n.levels && ch.ripl > 0 && ch.ripl < (int)n.levels->lv.size()) {
    const auto& L = n.levels->lv; int cur = ch.ripl;
    for (int guard = 0; guard < 80 && cur > 0 && cur < (int)L.size(); ++guard) {
      if (L[cur].br.empty()) { evs.push_back(L[cur].E * 1e6); break; }   // unknown decay: straight to ground state
      double u = rng.uni(); size_t j = 0;
      while (j + 1 < L[cur].br.size() && u >= L[cur].br[j].cum) ++j;
      const Branch& b = L[cur].br[j];
      if (rng.uni() < b.pgam) evs.push_back((L[cur].E - L[b.fin].E) * 1e6);   // else internal conversion
      cur = b.fin;
    }
  } else {
    evs.push_back(ch.Ex * 1e6);
  }
  emit_photons(T, bank, q, evs, false, rng);
}

static inline void tally_track(Tally& T, const Problem& P, const Particle& q, double len) {
  if (len <= 0) return;
  double x2 = q.x + q.mu * len, xa = std::min(q.x, x2), xb = std::max(q.x, x2);
  int i0 = std::min(P.nb - 1, std::max(0, (int)(xa / P.dx))), i1 = std::min(P.nb - 1, std::max(0, (int)(xb / P.dx)));
  double amu = std::fabs(q.mu);
  std::vector<double>* v = q.photon ? &T.flux_g : (q.E > 1e5 ? &T.flux_f : (q.E > E_COLD ? &T.flux_i : &T.flux_t));
  for (int i = i0; i <= i1; ++i) {
    double lo = std::max(xa, i * P.dx), hi = std::min(xb, (i + 1) * P.dx);
    if (hi > lo) (*v)[i] += q.w * (hi - lo) / amu;
  }
}
static inline int depth_bin(const Problem& P, double x) { return std::min(P.nb - 1, std::max(0, (int)(x / P.dx))); }

static double sample_source_E(const Problem& P, Rng& r) {
  for (;;) {
    double E;
    switch (P.src) {
      case Src::NMono: case Src::GMono: return P.E0_MeV * 1e6;
      case Src::NThermal: return 0.0253;
      case Src::Cs137: return 661657.0;
      case Src::Co60: return r.uni() < 0.5 ? 1173228.0 : 1332492.0;
      case Src::Cf252: E = sample_watt(r, 1.18, 1.03419) * 1e6; break;
      default: E = sample_watt(r, 0.988, 2.249) * 1e6; break;
    }
    if (E > 1e-3 && E < EMAX_N) return E;
  }
}

// Transport
static void run_batch(const Problem& P, uint64_t nhist, Rng& rng, Tally& T) {
  std::vector<Particle> bank; bank.reserve(64);
  const int nreg = (int)P.regions.size();
  const bool psrc = src_is_photon(P.src);
  XS xs_buf;

  for (uint64_t h = 0; h < nhist; ++h) {
    Particle p{0.0, P.cosine ? std::sqrt(rng.uni()) : 1.0, 0.0, 1.0, 0, psrc, false, false};
    p.E = sample_source_E(P, rng);
    T.nsrc += 1;
    const double inv_mu = 1.0 / std::max(p.mu, MU_MIN);
    T.dose_inc += (psrc ? P.dose_g[pbin(p.E)] : P.dose_n[dbin(p.E)]) * inv_mu;
    {
      double tau = 0, st, sb;
      for (auto& R : P.regions) {
        if (psrc) st = P.mats[R.mat].gst[pbin(p.E)]; else mlook(P.mats[R.mat], p.E, st, sb);
        tau += st * (R.x1 - R.x0);
      }
      T.unc_analytic += std::exp(-tau / std::max(p.mu, 1e-9));
    }
    bank.push_back(p);

    while (!bank.empty()) {
      Particle q = bank.back(); bank.pop_back();
      int k = q.region;
      for (;;) {
        const Region& R = P.regions[k];
        const Material& M = P.mats[R.mat];
        double st, sab = 0;
        if (q.photon) st = M.gst[pbin(q.E)]; else mlook(M, q.E, st, sab);
        const double d = -std::log(rng.uopen()) / st;
        if (std::fabs(q.mu) < 1e-9) q.mu = (q.mu < 0 ? -1e-9 : 1e-9);
        double db = (q.mu > 0 ? (R.x1 - q.x) : (R.x0 - q.x)) / q.mu;
        if (db < 0) db = 0;
        tally_track(T, P, q, std::min(d, db));

        if (d >= db) {  // surface crossing ----
          int kn = q.mu > 0 ? k + 1 : k - 1;
          q.x = q.mu > 0 ? R.x1 : R.x0;
          if (kn < 0) {
            if (q.photon) { T.g_refl += q.w; T.sp_gr[spec_bin(q.E, SPEC_LO_P, SPEC_HI_P)] += q.w; }
            else { T.n_refl += q.w; T.sp_nr[spec_bin(q.E, SPEC_LO_N, SPEC_HI_N)] += q.w; }
            break;
          }
          if (kn >= nreg) {
            double iw = q.w / std::max(std::fabs(q.mu), MU_MIN);
            if (q.photon) {
              T.g_trans += q.w; if (!q.collided && !q.secondary) T.g_unc += q.w;
              T.dose_g += P.dose_g[pbin(q.E)] * iw; T.sp_gt[spec_bin(q.E, SPEC_LO_P, SPEC_HI_P)] += q.w;
            } else {
              T.n_trans += q.w; if (!q.collided) T.n_unc += q.w;
              T.dose_n += P.dose_n[dbin(q.E)] * iw; T.sp_nt[spec_bin(q.E, SPEC_LO_N, SPEC_HI_N)] += q.w;
            }
            break;
          }
          double ratio = P.regions[kn].imp / R.imp;
          if (ratio > 1.0) {
            int n = (int)ratio; if (rng.uni() < ratio - n) ++n;
            q.w /= ratio; q.region = kn;
            for (int c = 1; c < n; ++c) bank.push_back(q);
          } else if (ratio < 1.0) {
            if (rng.uni() < ratio) q.w /= ratio; else break;
          }
          k = kn; q.region = kn;
          continue;
        }

        // collision ----
        q.x += q.mu * d; q.collided = true;

        if (q.photon) {
          const int b = pbin(q.E);
          const double* c = &M.g_cum[b * 3];
          double u = rng.uni() * st;
          if (u < c[0]) { T.g_abs += q.w; break; }
          if (u < c[1]) {
            double al = q.E / ME_EV, eps, r1, r2, r3;
            do {
              r1 = rng.uni(); r2 = rng.uni(); r3 = rng.uni();
              if (r1 <= (1 + 2 * al) / (9 + 2 * al)) { eps = 1 + 2 * al * r2; if (r3 <= 4 * (1 / eps - 1 / (eps * eps))) break; }
              else { eps = (1 + 2 * al) / (1 + 2 * al * r2); double t = 1 - (eps - 1) / al; if (r3 <= 0.5 * (t * t + 1 / eps)) break; }
            } while (true);
            double mul = 1 - (eps - 1) / al;
            q.E /= eps; q.mu = rotate_mu(rng, q.mu, mul);
            if (q.E < EMIN_P) { T.g_abs += q.w; break; }
          } else {
            T.pair_extra += q.w;
            double mi = 2 * rng.uni() - 1;
            Particle o = q; o.E = ME_EV; o.mu = -mi; bank.push_back(o);
            q.E = ME_EV; q.mu = mi;
          }
          if (q.w < P.w_low) { if (rng.uni() < q.w / P.w_surv) q.w = P.w_surv; else break; }
          continue;
        }

        // neutron collision
        const double pa = std::min(1.0, sab / st);
        const bool ab_event = rng.uni() < pa;
        const size_t niso = M.nuc.size();
        if (ab_event && !P.no_gamma) {            // capture photons are sampled without bias, even with implicit capture
          double wsum[64]; double tot = 0;
          for (size_t i = 0; i < niso; ++i) {
            double sc_, ab_; iso_sc_ab(*NUC[M.nuc[i]], M.tsl[i] >= 0 ? TSLS[M.tsl[i]].get() : nullptr, q.E, sc_, ab_);
            tot += wsum[i] = M.dens[i] * ab_;
          }
          double u = rng.uni() * tot; size_t i = 0;
          while (i + 1 < niso && u >= wsum[i]) { u -= wsum[i]; ++i; }
          iso_xs(*NUC[M.nuc[i]], M.tsl[i] >= 0 ? TSLS[M.tsl[i]].get() : nullptr, q.E, xs_buf);
          int type = (rng.uni() * xs_buf.abs < xs_buf.cap) ? C_CAP : C_OABS;
          capture_photons(T, bank, q, *NUC[M.nuc[i]], type, rng);
        }
        if (P.analog || q.E < E_ANALOG) {
          if (ab_event) { T.n_abs += q.w; T.absprof[depth_bin(P, q.x)] += q.w; break; }
        } else {
          double wa = q.w * pa; T.n_abs += wa; T.absprof[depth_bin(P, q.x)] += wa; q.w *= (1 - pa);
          if (pa >= 1.0) break;
        }

        // choose scattering isotope
        double wsc[64]; double tot = 0;
        for (size_t i = 0; i < niso; ++i) {
          double sc_, ab_; iso_sc_ab(*NUC[M.nuc[i]], M.tsl[i] >= 0 ? TSLS[M.tsl[i]].get() : nullptr, q.E, sc_, ab_);
          tot += wsc[i] = M.dens[i] * sc_;
        }
        if (tot <= 0) break;
        double u = rng.uni() * tot; size_t ii = 0;
        while (ii + 1 < niso && u >= wsc[ii]) { u -= wsc[ii]; ++ii; }
        const Nuc& n = *NUC[M.nuc[ii]]; const TSL* tsl = M.tsl[ii] >= 0 ? TSLS[M.tsl[ii]].get() : nullptr;
        iso_xs(n, tsl, q.E, xs_buf);
        // choose channel
        double v = rng.uni() * xs_buf.scat; int ch = -1; int ty = C_EL;
        if (v < xs_buf.el) ty = C_EL;
        else {
          v -= xs_buf.el; ty = -1;
          for (int l = 0; l < xs_buf.nlev; ++l) { if (v < xs_buf.lev[l]) { ty = C_LEV; ch = l; break; } v -= xs_buf.lev[l]; }
          if (ty < 0) ty = (v < xs_buf.n2n) ? C_N2N : C_CONT;
        }

        if (ty == C_EL) {
          if (tsl && q.E < tsl->emax) {
            double Eo, mul; tsl_scatter(*tsl, rng, q.E, Eo, mul); q.E = Eo; q.mu = rotate_mu(rng, q.mu, mul);
          } else if (q.E < FG_THRESH * KT) {
            elastic_free_gas(rng, q.E, q.mu, n.A);
          } else {
            double Eo, mul; two_body(rng, q.E, n.A, 0.0, Eo, mul); q.E = Eo; q.mu = rotate_mu(rng, q.mu, mul);
          }
        } else if (ty == C_LEV) {
          const Chan& lc = n.lev[ch];
          double Ex = std::min(lc.Ex, 0.98 * q.E * 1e-6 * n.A / (n.A + 1));
          double Eo, mul; two_body(rng, q.E, n.A, Ex, Eo, mul);
          if (!P.no_gamma) level_cascade(T, bank, q, n, lc, rng);
          q.E = Eo; q.mu = rotate_mu(rng, q.mu, mul);
        } else if (ty == C_CONT) {
          double Em = q.E * 1e-6, thr = n.cont.xs.empty() ? 0.0 : n.E[std::min<size_t>(n.cont.start, n.E.size() - 1)] * 1e-6;
          double Eav = std::max(Em - thr, 0.05 * Em);
          double Tn = std::max(0.2, std::sqrt(8.0 * Eav / n.A));
          double En = sample_evap(rng, Eav, Tn);
          if (!P.no_gamma) {
            double Eg = std::max(0.0, Eav - En);
            int ng = std::max(1, std::min(6, (int)std::ceil(Eg / 2.0)));
            if (Eg > 0.01) emit_photons(T, bank, q, std::vector<double>(ng, Eg * 1e6 / ng), false, rng);
          }
          q.E = std::max(En * 1e6, 1e-3); q.mu = 2 * rng.uni() - 1;
        } else {  // (n,2n)
          double Em = q.E * 1e-6, thr = n.E[std::min<size_t>(n.n2n.start, n.E.size() - 1)] * 1e-6 * n.A / (n.A + 1);
          double Eav = std::max(Em - thr, 0.05 * Em);
          double Tn = std::max(0.2, std::sqrt(8.0 * Eav / n.A));
          double E1 = sample_evap(rng, Eav, Tn), E2 = sample_evap(rng, std::max(Eav - E1, 1e-3), Tn);
          T.n2n_extra += q.w;
          Particle s = q; s.E = std::max(E2 * 1e6, 1e-3); s.mu = 2 * rng.uni() - 1; bank.push_back(s);
          q.E = std::max(E1 * 1e6, 1e-3); q.mu = 2 * rng.uni() - 1;
        }
        if (q.w < P.w_low) { if (rng.uni() < q.w / P.w_surv) q.w = P.w_surv; else break; }
      }
    }
  }
}

// CLI / output
static void usage() {
  std::fprintf(stderr,
    "1DMC-N -- 1D coupled neutron/photon Monte Carlo slab shielding (educational)\n\n"
    "  1dmc-n --layer water:20 --layer lead:5 --source cf252 --n 1000000\n\n"
    "Options:\n"
    "  --layer MAT:CM          add a layer (repeat; first layer faces the source)\n"
    "  --material N:RHO:H=w,.. custom material (mass fractions of elements or isotopes, e.g. B10)\n"
    "  --tsl MAT:ISO=TABLE     thermal scattering table for an isotope in a material, e.g. mymat:H1=hpoly\n"
    "  --no-tsl                disable S(alpha,beta) tables (free-gas hydrogen everywhere)\n"
    "  --source mono|cf252|u235|thermal|gamma|cs137|co60   (default mono = mono-energetic neutrons)\n"
    "  --energy MeV            energy for mono / gamma source (default 2.0)\n"
    "  --angle beam|cosine     normal beam or isotropic incidence (default beam)\n"
    "  --n N                   histories (default 1e6)\n"
    "  --batches B             batches for statistics / threads (default 10)\n"
    "  --bins NB               depth bins for flux profile (default 50)\n"
    "  --split R               importance ratio per cell, >1 enables splitting (default 1)\n"
    "  --cell CM               cell thickness for splitting (default total/20)\n"
    "  --analog                disable implicit capture\n"
    "  --no-gamma              do not produce secondary photons\n"
    "  --generic-gamma         ignore EGAF/RIPL data: generic capture cascade, one photon per inelastic level\n"
    "  --dose GEOM             ICRP-116 geometry: AP PA LLAT RLAT ROT ISO (default AP)\n"
    "  --data DIR              data directory (default: ../data next to the binary, or $ONEDMC_N_DATA)\n"
    "  --seed S                RNG seed (default 12345)\n"
    "  --list-materials        print built-in materials as JSON\n");
}
static void print_arr(std::ostringstream& o, const std::vector<double>& v) {
  o << "[";
  for (size_t i = 0; i < v.size(); ++i) { if (i) o << ","; char b[40]; std::snprintf(b, sizeof b, "%.6e", v[i]); o << b; }
  o << "]";
}
static void print_stat(std::ostringstream& o, const std::vector<double>& pb) {
  double n = (double)pb.size(), m = 0, v = 0;
  for (double x : pb) m += x;
  m /= n;
  for (double x : pb) v += (x - m) * (x - m);
  double se = n > 1 ? std::sqrt(v / (n - 1) / n) : 0.0;
  char b[120]; std::snprintf(b, sizeof b, "{\"mean\":%.6e,\"stderr\":%.6e}", m, se); o << b;
}

int main(int argc, char** argv) try {
  Problem P;
  std::vector<MatDef> defs = builtin_materials();
  auto tsl_map = default_tsl();
  std::vector<std::pair<std::string, double>> layer_req;
  uint64_t N = 1000000, seed = 12345; int B = 10; double cell = 0; bool list = false, no_tsl = false;
  std::string data_hint, geom = "AP", srcname = "mono", angle = "beam";

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* what) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + what);
      return argv[++i];
    };
    if (a == "--help" || a == "-h") { usage(); return 0; }
    else if (a == "--list-materials") list = true;
    else if (a == "--layer") {
      auto p = split(need("--layer"), ':');
      if (p.size() != 2) throw std::runtime_error("--layer expects MATERIAL:THICKNESS_CM");
      layer_req.push_back({lower(p[0]), std::atof(p[1].c_str())});
    } else if (a == "--material") {
      auto p = split(need("--material"), ':');
      if (p.size() != 3) throw std::runtime_error("--material expects NAME:RHO:Nuc=frac,Nuc=frac");
      MatDef d; d.name = lower(p[0]); d.rho = std::atof(p[1].c_str());
      for (auto& kv : split(p[2], ',')) {
        auto q = split(kv, '=');
        if (q.size() != 2) throw std::runtime_error("bad composition entry '" + kv + "'");
        d.mass_frac.push_back({q[0], std::atof(q[1].c_str())});
      }
      defs.push_back(d);
    } else if (a == "--tsl") {
      auto p = split(need("--tsl"), ':');
      if (p.size() != 2) throw std::runtime_error("--tsl expects MATERIAL:ISOTOPE=TABLE");
      auto q = split(p[1], '=');
      if (q.size() != 2) throw std::runtime_error("--tsl expects MATERIAL:ISOTOPE=TABLE");
      tsl_map[lower(p[0])].push_back({q[0], q[1]});
    } else if (a == "--no-tsl") no_tsl = true;
    else if (a == "--source") srcname = lower(need("--source"));
    else if (a == "--energy") P.E0_MeV = std::atof(need("--energy").c_str());
    else if (a == "--angle") angle = lower(need("--angle"));
    else if (a == "--n") N = (uint64_t)std::atof(need("--n").c_str());
    else if (a == "--batches") B = std::max(2, std::atoi(need("--batches").c_str()));
    else if (a == "--bins") P.nb = std::max(1, std::atoi(need("--bins").c_str()));
    else if (a == "--split") P.split = std::max(1.0, std::atof(need("--split").c_str()));
    else if (a == "--cell") cell = std::atof(need("--cell").c_str());
    else if (a == "--analog") P.analog = true;
    else if (a == "--no-gamma") P.no_gamma = true;
    else if (a == "--generic-gamma") USE_DETAILED_GAMMA = false;
    else if (a == "--dose") geom = need("--dose");
    else if (a == "--data") data_hint = need("--data");
    else if (a == "--seed") seed = (uint64_t)std::atoll(need("--seed").c_str());
    else { usage(); throw std::runtime_error("unknown option " + a); }
  }

  if (list) {
    std::ostringstream o; o << "[";
    for (size_t i = 0; i < defs.size(); ++i) {
      if (i) o << ",";
      o << "{\"name\":\"" << defs[i].name << "\",\"rho\":" << defs[i].rho << ",\"composition\":{";
      for (size_t k = 0; k < defs[i].mass_frac.size(); ++k)
        o << (k ? "," : "") << "\"" << defs[i].mass_frac[k].first << "\":" << defs[i].mass_frac[k].second;
      o << "}}";
    }
    o << "]"; std::puts(o.str().c_str());
    return 0;
  }

  if (srcname == "mono") P.src = Src::NMono; else if (srcname == "cf252") P.src = Src::Cf252;
  else if (srcname == "u235") P.src = Src::U235; else if (srcname == "thermal") P.src = Src::NThermal;
  else if (srcname == "gamma") P.src = Src::GMono; else if (srcname == "cs137") P.src = Src::Cs137;
  else if (srcname == "co60") P.src = Src::Co60; else throw std::runtime_error("unknown source '" + srcname + "'");
  P.cosine = (angle == "cosine");
  if (P.E0_MeV <= 0 || P.E0_MeV > 20) throw std::runtime_error("--energy must be in (0, 20] MeV");
  if (P.src == Src::GMono && P.E0_MeV * 1e6 < EMIN_P) throw std::runtime_error("photon energy must be >= 0.01 MeV");
  if (N < (uint64_t)B) throw std::runtime_error("--n must be >= --batches");

  DATA_DIR = find_data_dir(data_hint, argv[0]);
  load_tables();
  NUC.reserve(ISOTOPES.size()); TSLS.reserve(16);

  {
    DoseTable dn = load_dose("icrp116_neutron.csv", geom), dg = load_dose("icrp116_photon.csv", geom);
    P.dose_n.resize(NDOSE); P.dose_g.resize(NB_P);
    for (int b = 0; b < NDOSE; ++b) P.dose_n[b] = dose_interp(dn, std::exp(LN_DLO + (b + 0.5) * DLN_D) * 1e-6);
    for (int b = 0; b < NB_P; ++b) P.dose_g[b] = dose_interp(dg, std::exp(LN_EMIN_P + (b + 0.5) * DLN_P) * 1e-6);
  }

  std::map<std::string, int> built;
  P.mats.reserve(layer_req.size() + 1);
  for (auto& lr : layer_req) {
    if (lr.second <= 0) continue;
    if (!built.count(lr.first)) {
      const MatDef* d = nullptr;
      for (auto& x : defs) if (x.name == lr.first) d = &x;
      if (!d) throw std::runtime_error("unknown material '" + lr.first + "'");
      std::vector<std::pair<std::string, std::string>> tr;
      if (!no_tsl && tsl_map.count(lr.first)) tr = tsl_map[lr.first];
      P.mats.push_back(build_material(*d, tr)); built[lr.first] = (int)P.mats.size() - 1;
      P.mats.back().uh.build(P.mats.back().U);            // hash must point at the element's own grid
    }
    P.layers.push_back(lr); P.layer_mat.push_back(built[lr.first]); P.total += lr.second;
  }
  if (P.layers.empty()) throw std::runtime_error("no layers with positive thickness (use --layer MAT:CM)");
  P.dx = P.total / P.nb;

  double x = 0, cs = (cell > 0 ? cell : P.total / 20.0);
  for (size_t l = 0; l < P.layers.size(); ++l) {
    double t = P.layers[l].second;
    int n = P.split > 1.0 ? std::max(1, (int)std::ceil(t / cs - 1e-9)) : 1;
    for (int s = 0; s < n; ++s) {
      Region R; R.x0 = x + t * s / n; R.x1 = x + t * (s + 1) / n; R.mat = P.layer_mat[l];
      R.imp = std::pow(P.split, (double)P.regions.size()); P.regions.push_back(R);
    }
    x += t;
  }
  if (P.regions.size() > 2000) throw std::runtime_error("too many splitting cells; increase --cell");

  auto t0 = std::chrono::steady_clock::now();
  std::vector<Tally> tal(B, Tally(P.nb));
  std::vector<uint64_t> nh(B, N / B);
  for (uint64_t r = 0; r < N % B; ++r) nh[r]++;
#pragma omp parallel for schedule(dynamic)
  for (int b = 0; b < B; ++b) {
    Rng rng(seed * 1000003ULL + (uint64_t)b * 7919ULL + 1);
    run_batch(P, nh[b], rng, tal[b]);
  }
  double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  Tally S(P.nb);
  std::vector<double> bnt, bnr, bna, bnu, bgt, bgr, bga, bgu, bdn, bdg, bdt;
  auto add = [](std::vector<double>& a, const std::vector<double>& b) { for (size_t i = 0; i < a.size(); ++i) a[i] += b[i]; };
  for (auto& t : tal) {
    S.nsrc += t.nsrc; S.n_trans += t.n_trans; S.n_refl += t.n_refl; S.n_abs += t.n_abs; S.n_unc += t.n_unc; S.n2n_extra += t.n2n_extra;
    S.g_trans += t.g_trans; S.g_refl += t.g_refl; S.g_abs += t.g_abs; S.g_unc += t.g_unc; S.pair_extra += t.pair_extra;
    S.dose_inc += t.dose_inc; S.dose_n += t.dose_n; S.dose_g += t.dose_g; S.unc_analytic += t.unc_analytic;
    S.prod_cap += t.prod_cap; S.prod_inel += t.prod_inel; S.prod_cap_E += t.prod_cap_E; S.prod_inel_E += t.prod_inel_E;
    S.cap_events += t.cap_events; S.cap_detailed += t.cap_detailed;
    add(S.flux_f, t.flux_f); add(S.flux_i, t.flux_i); add(S.flux_t, t.flux_t); add(S.flux_g, t.flux_g); add(S.absprof, t.absprof);
    add(S.sp_nt, t.sp_nt); add(S.sp_nr, t.sp_nr); add(S.sp_gt, t.sp_gt); add(S.sp_gr, t.sp_gr); add(S.sp_gp, t.sp_gp);
    double n = t.nsrc;
    bnt.push_back(t.n_trans / n); bnr.push_back(t.n_refl / n); bna.push_back(t.n_abs / n); bnu.push_back(t.n_unc / n);
    bgt.push_back(t.g_trans / n); bgr.push_back(t.g_refl / n); bga.push_back(t.g_abs / n); bgu.push_back(t.g_unc / n);
    bdn.push_back(t.dose_n / t.dose_inc); bdg.push_back(t.dose_g / t.dose_inc); bdt.push_back((t.dose_n + t.dose_g) / t.dose_inc);
  }
  const double ns = S.nsrc;
  const bool psrc = src_is_photon(P.src);

  std::ostringstream o; char buf[400];
  o << "{\"schema\":3,\"source\":{\"type\":\"" << srcname << "\",\"particle\":\"" << (psrc ? "photon" : "neutron")
    << "\",\"energy_MeV\":" << P.E0_MeV << ",\"angle\":\"" << angle << "\"}"
    << ",\"n_particles\":" << (unsigned long long)N << ",\"batches\":" << B << ",\"threads\":"
#ifdef _OPENMP
    << omp_get_max_threads()
#else
    << 1
#endif
    << ",\"runtime_s\":" << secs << ",\"total_thickness_cm\":" << P.total
    << ",\"dose_quantity\":\"ICRP-116 effective dose, " << geom << " geometry\""
    << ",\"physics\":{\"neutron_data\":\"ENDF/B-VIII.0 pointwise 294 K\",\"gamma_model\":\""
    << (USE_DETAILED_GAMMA ? "EGAF capture cascades + RIPL level cascades" : "generic") << "\",\"thermal\":[";
  { bool first = true;
    for (auto& m : P.mats) for (auto& nt : m.notes) { o << (first ? "" : ",") << "\"" << nt << "\""; first = false; } }
  o << "]},\"layers\":[";
  double xx = 0;
  for (size_t l = 0; l < P.layers.size(); ++l) {
    std::snprintf(buf, sizeof buf, "%s{\"material\":\"%s\",\"thickness_cm\":%g,\"x0\":%g,\"x1\":%g}",
                  l ? "," : "", P.layers[l].first.c_str(), P.layers[l].second, xx, xx + P.layers[l].second);
    o << buf; xx += P.layers[l].second;
  }
  o << "],\"neutron\":{\"transmitted\":"; print_stat(o, bnt);
  o << ",\"reflected\":"; print_stat(o, bnr); o << ",\"absorbed\":"; print_stat(o, bna);
  o << ",\"uncollided_transmitted\":"; print_stat(o, bnu);
  std::snprintf(buf, sizeof buf, ",\"multiplication_extra\":%.6e", S.n2n_extra / ns); o << buf;
  o << "},\"photon\":{\"transmitted\":"; print_stat(o, bgt);
  o << ",\"reflected\":"; print_stat(o, bgr); o << ",\"absorbed\":"; print_stat(o, bga);
  o << ",\"uncollided_transmitted\":"; print_stat(o, bgu);
  std::snprintf(buf, sizeof buf, ",\"pair_extra\":%.6e,\"produced_from_capture\":%.6e,\"produced_from_inelastic\":%.6e,"
                "\"produced_energy_MeV_capture\":%.6e,\"produced_energy_MeV_inelastic\":%.6e,"
                "\"capture_events\":%.6e,\"capture_measured_cascade_fraction\":%.4f}",
                S.pair_extra / ns, S.prod_cap / ns, S.prod_inel / ns, S.prod_cap_E / ns, S.prod_inel_E / ns,
                S.cap_events / ns, S.cap_events > 0 ? S.cap_detailed / S.cap_events : 0.0);
  o << buf;
  std::snprintf(buf, sizeof buf, ",\"uncollided_analytic\":%.6e", S.unc_analytic / ns); o << buf;
  std::snprintf(buf, sizeof buf, ",\"dose\":{\"incident_pSv_cm2\":%.6e,\"neutron_ratio\":", S.dose_inc / ns); o << buf;
  print_stat(o, bdn); o << ",\"photon_ratio\":"; print_stat(o, bdg); o << ",\"total_ratio\":"; print_stat(o, bdt); o << "}";

  std::vector<double> edges, f1(P.nb), f2(P.nb), f3(P.nb), f4(P.nb), ab(P.nb), se_n, se_p, v1, v2, v3, v4, v5;
  for (int i = 0; i <= P.nb; ++i) edges.push_back(i * P.dx);
  for (int i = 0; i < P.nb; ++i) {
    f1[i] = S.flux_f[i] / (P.dx * ns); f2[i] = S.flux_i[i] / (P.dx * ns); f3[i] = S.flux_t[i] / (P.dx * ns);
    f4[i] = S.flux_g[i] / (P.dx * ns); ab[i] = S.absprof[i] / (P.dx * ns);
  }
  o << ",\"depth_edges_cm\":"; print_arr(o, edges);
  o << ",\"flux_depth\":{\"neutron_fast\":"; print_arr(o, f1); o << ",\"neutron_intermediate\":"; print_arr(o, f2);
  o << ",\"neutron_thermal\":"; print_arr(o, f3); o << ",\"photon\":"; print_arr(o, f4); o << "}";
  o << ",\"absorption_depth\":"; print_arr(o, ab);
  for (int i = 0; i <= NSPEC; ++i) {
    se_n.push_back(SPEC_LO_N * std::pow(SPEC_HI_N / SPEC_LO_N, (double)i / NSPEC));
    se_p.push_back(SPEC_LO_P * std::pow(SPEC_HI_P / SPEC_LO_P, (double)i / NSPEC));
  }
  for (int i = 0; i < NSPEC; ++i) { v1.push_back(S.sp_nt[i] / ns); v2.push_back(S.sp_nr[i] / ns); v3.push_back(S.sp_gt[i] / ns); v4.push_back(S.sp_gr[i] / ns); v5.push_back(S.sp_gp[i] / ns); }
  o << ",\"neutron_spectrum_edges_eV\":"; print_arr(o, se_n);
  o << ",\"neutron_spectrum_transmitted\":"; print_arr(o, v1); o << ",\"neutron_spectrum_reflected\":"; print_arr(o, v2);
  o << ",\"photon_spectrum_edges_eV\":"; print_arr(o, se_p);
  o << ",\"photon_spectrum_transmitted\":"; print_arr(o, v3); o << ",\"photon_spectrum_reflected\":"; print_arr(o, v4);
  o << ",\"photon_spectrum_produced\":"; print_arr(o, v5);
  o << "}";
  std::puts(o.str().c_str());
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
