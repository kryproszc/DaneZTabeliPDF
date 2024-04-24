#ifndef CSV_DEFS
#define CSV_DEFS

#define CSV_PADDING 64

#endif


///////


#ifndef SIMDCSV_COMMON_DEFS_H
#define SIMDCSV_COMMON_DEFS_H

#include <cassert>

#ifdef __AVX2__
#define SIMDCSV_PADDING  sizeof(__m256i)
#else

#define SIMDCSV_PADDING  32
#endif

#define ROUNDUP_N(a, n) (((a) + ((n)-1)) & ~((n)-1))
#define ROUNDDOWN_N(a, n) ((a) & ~((n)-1))

#define ISALIGNED_N(ptr, n) (((uintptr_t)(ptr) & ((n)-1)) == 0)

#ifdef _MSC_VER

#define really_inline inline
#define never_inline __declspec(noinline)

#define UNUSED
#define WARN_UNUSED

#ifndef likely
#define likely(x) x
#endif
#ifndef unlikely
#define unlikely(x) x
#endif

#else

#define really_inline inline __attribute__((always_inline, unused))
#define never_inline inline __attribute__((noinline, unused))

#define UNUSED __attribute__((unused))
#define WARN_UNUSED __attribute__((warn_unused_result))

#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif
#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

#endif  

#endif 


/////


#ifndef TIMING_H
#define TIMING_H

#ifdef __linux__
#include <asm/unistd.h>       
#include <linux/perf_event.h> 
#include <sys/ioctl.h>        
#include <unistd.h>           

#include <cerrno>  
#include <cstring> 
#include <stdexcept>
#include <vector>
#include <iostream>

class TimingAccumulator {
public:
  std::vector<uint64_t> results;
  std::vector<uint64_t> temp_result_vec; 
  int num_phases;
  int num_events;
  int fd;
  bool working;
  
  explicit TimingAccumulator(int num_phases_in, std::vector<int> config_vec) 
    : num_phases(num_phases_in), fd(0), working(true) {
    perf_event_attr attribs;
    std::vector<uint64_t> ids;
    
    memset(&attribs, 0, sizeof(attribs));
    attribs.type = PERF_TYPE_HARDWARE; 
    attribs.size = sizeof(attribs);
    attribs.disabled = 1;
    attribs.exclude_kernel = 1;
    attribs.exclude_hv = 1;
    
    attribs.sample_period = 0;
    attribs.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
    const int pid = 0;  
    const int cpu = -1; 
    const unsigned long flags = 0;
    
    int group = -1; 
    num_events = config_vec.size();
    ids.resize(config_vec.size());
    uint32_t i = 0;
    for (auto config : config_vec) {
      attribs.config = config;
      fd = syscall(__NR_perf_event_open, &attribs, pid, cpu, group, flags);
      if (fd == -1) {
        report_error("perf_event_open");
      }
      ioctl(fd, PERF_EVENT_IOC_ID, &ids[i++]);
      if (group == -1) {
        group = fd;
      }
    }
    
    temp_result_vec.resize(num_events * 2 + 1);
    results.resize(num_phases*num_events, config_vec.size());
  }
  
  ~TimingAccumulator() {
    close(fd);
  }
  
  void report_error(const std::string &context) {
    if (working) {
      std::cerr << (context + ": " + std::string(strerror(errno))) << std::endl;
    }
    working = false;
  }
  
  void start(UNUSED int phase_number) {
    if (ioctl(fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) == -1) {
      report_error("ioctl(PERF_EVENT_IOC_RESET)");
    }
    if (ioctl(fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) == -1) {
      report_error("ioctl(PERF_EVENT_IOC_ENABLE)");
    }
  }
  
  void stop(int phase_number) {
    if (ioctl(fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) == -1) {
      report_error("ioctl(PERF_EVENT_IOC_DISABLE)");
    }
    
    if (read(fd, temp_result_vec.data(), temp_result_vec.size() * 8) == -1) {
      report_error("read");
    }
    
    for (uint32_t i = 1; i < temp_result_vec.size(); i += 2) {
      results[phase_number * num_events + i/2] += temp_result_vec[i];
    }
  }
  
  void dump() {
    for (int i = 0; i < num_phases; i++) {
      for (int j = 0; j < num_events; j++) {
        std::cout << results[i*num_events + j] << " ";
      }
      std::cout << "\n";
    }
  }
};

class TimingPhase {
public:
  TimingAccumulator & acc;
  int phase_number;
  
  TimingPhase(TimingAccumulator & acc_in, int phase_number_in) 
    : acc(acc_in), phase_number(phase_number_in) {
    acc.start(phase_number);
  }
  
  ~TimingPhase() {
    acc.stop(phase_number);
  }
};
#endif 

#endif



/////

#ifndef PORTABILITY_H
#define PORTABILITY_H

#ifdef _MSC_VER

#include <intrin.h>
#include <iso646.h>
#include <cstdint>

static inline bool add_overflow(uint64_t value1, uint64_t value2, uint64_t *result) {
  return _addcarry_u64(0, value1, value2, reinterpret_cast<unsigned __int64 *>(result));
}

#pragma intrinsic(_umul128)
static inline bool mul_overflow(uint64_t value1, uint64_t value2, uint64_t *result) {
  uint64_t high;
  *result = _umul128(value1, value2, &high);
  return high;
}

static inline int trailingzeroes(uint64_t input_num) {
  return _tzcnt_u64(input_num);
}

static inline int leadingzeroes(uint64_t  input_num) {
  return _lzcnt_u64(input_num);
}

static inline int hamming(uint64_t input_num) {
#ifdef _WIN64  
  return (int)__popcnt64(input_num);
#else  
  return (int)(__popcnt((uint32_t)input_num) +
          __popcnt((uint32_t)(input_num >> 32)));
#endif
}

#else
#include <cstdint>
#include <cstdlib>

#if defined(__BMI2__) || defined(__POPCOUNT__) || defined(__AVX2__)
#include <x86intrin.h>
#endif

static inline bool add_overflow(uint64_t  value1, uint64_t  value2, uint64_t *result) {
  return __builtin_uaddll_overflow(value1, value2, (unsigned long long*)result);
}
static inline bool mul_overflow(uint64_t  value1, uint64_t  value2, uint64_t *result) {
  return __builtin_umulll_overflow(value1, value2, (unsigned long long *)result);
}

static inline int trailingzeroes(uint64_t input_num) {
#ifdef __BMI2__
  return _tzcnt_u64(input_num);
#else
  return __builtin_ctzll(input_num);
#endif
}

static inline int leadingzeroes(uint64_t  input_num) {
#ifdef __BMI2__
  return _lzcnt_u64(input_num);
#else
  return __builtin_clzll(input_num);
#endif
}

static inline int hamming(uint64_t input_num) {
#ifdef __POPCOUNT__
  return _popcnt64(input_num);
#else
  return __builtin_popcountll(input_num);
#endif
}

#endif 

#endif 



//////

#ifndef MEM_UTIL_H
#define MEM_UTIL_H

#include <stdlib.h>
#include <cstdint>

static inline void *aligned_malloc(size_t alignment, size_t size) {
  void *p;
#ifdef _MSC_VER
  p = _aligned_malloc(size, alignment);
#elif defined(__MINGW32__) || defined(__MINGW64__)
  p = __mingw_aligned_malloc(size, alignment);
#else
  
  if (posix_memalign(&p, alignment, size) != 0) { return nullptr; }
#endif
  return p;
}

static inline void aligned_free(void *memblock) {
  if(memblock == nullptr) { return; }
#ifdef _MSC_VER
  _aligned_free(memblock);
#elif defined(__MINGW32__) || defined(__MINGW64__)
  __mingw_aligned_free(memblock);
#else
  free(memblock);
#endif
}

#endif



///////


#include <unistd.h> 

#include <iostream>
#include <vector>

#include "common_defs.h"
#include "csv_defs.h"
#include "io_util.h"
#include "timing.h"
#include "mem_util.h"
#include "portability.h"
using namespace std;

struct ParsedCSV {
  uint32_t n_indexes{0};
  uint32_t *indexes; 
};

struct simd_input {
#ifdef __AVX2__
  __m256i lo;
  __m256i hi;
#else
#error "Problem z AVX"
#endif
};

really_inline simd_input fill_input(const uint8_t * ptr) {
  struct simd_input in;
#ifdef __AVX2__
  in.lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr + 0));
  in.hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr + 32));
  return in;
}
#endif

really_inline uint64_t cmp_mask_against_input(simd_input in, uint8_t m) {
#ifdef __AVX2__
  const __m256i mask = _mm256_set1_epi8(m);
  __m256i cmp_res_0 = _mm256_cmpeq_epi8(in.lo, mask);
  uint64_t res_0 = static_cast<uint32_t>(_mm256_movemask_epi8(cmp_res_0));
  __m256i cmp_res_1 = _mm256_cmpeq_epi8(in.hi, mask);
  uint64_t res_1 = _mm256_movemask_epi8(cmp_res_1);
  return res_0 | (res_1 << 32);
#endif
}

really_inline uint64_t find_quote_mask(simd_input in, uint64_t &prev_iter_inside_quote) {
  uint64_t quote_bits = cmp_mask_against_input(in, '"');
  
#ifdef __AVX2__
  uint64_t quote_mask = _mm_cvtsi128_si64(_mm_clmulepi64_si128(
    _mm_set_epi64x(0ULL, quote_bits), _mm_set1_epi8(0xFF), 0));
#elif defined(__ARM_NEON)
  uint64_t quote_mask = vmull_p64( -1ULL, quote_bits);
#endif
  quote_mask ^= prev_iter_inside_quote;
  
  prev_iter_inside_quote =
    static_cast<uint64_t>(static_cast<int64_t>(quote_mask) >> 63);
  return quote_mask;
}

really_inline void flatten_bits(uint32_t *base_ptr, uint32_t &base,
                                uint32_t idx, uint64_t bits) {
  if (bits != 0u) {
    uint32_t cnt = hamming(bits);
    uint32_t next_base = base + cnt;
    base_ptr[base + 0] = static_cast<uint32_t>(idx) + trailingzeroes(bits);
    bits = bits & (bits - 1);
    base_ptr[base + 1] = static_cast<uint32_t>(idx) + trailingzeroes(bits);
    bits = bits & (bits - 1);
    base_ptr[base + 2] = static_cast<uint32_t>(idx) + trailingzeroes(bits);
    bits = bits & (bits - 1);
    base_ptr[base + 3] = static_cast<uint32_t>(idx) + trailingzeroes(bits);
    bits = bits & (bits - 1);
    base_ptr[base + 4] = static_cast<uint32_t>(idx) + trailingzeroes(bits);
    bits = bits & (bits - 1);
    base_ptr[base + 5] = static_cast<uint32_t>(idx) + trailingzeroes(bits);
    bits = bits & (bits - 1);
    base_ptr[base + 6] = static_cast<uint32_t>(idx) + trailingzeroes(bits);
    bits = bits & (bits - 1);
    base_ptr[base + 7] = static_cast<uint32_t>(idx) + trailingzeroes(bits);
    bits = bits & (bits - 1);
    if (cnt > 8) {
      base_ptr[base + 8] = static_cast<uint32_t>(idx) + trailingzeroes(bits);
      bits = bits & (bits - 1);
      base_ptr[base + 9] = static_cast<uint32_t>(idx) + trailingzeroes(bits);
      bits = bits & (bits - 1);
      base_ptr[base + 10] = static_cast<uint32_t>(idx) + trailingzeroes(bits);
      bits = bits & (bits - 1);
      base_ptr[base + 11] = static_cast<uint32_t>(idx) + trailingzeroes(bits);
      bits = bits & (bits - 1);
      base_ptr[base + 12] = static_cast<uint32_t>(idx) + trailingzeroes(bits);
      bits = bits & (bits - 1);
      base_ptr[base + 13] = static_cast<uint32_t>(idx) + trailingzeroes(bits);
      bits = bits & (bits - 1);
      base_ptr[base + 14] = static_cast<uint32_t>(idx) + trailingzeroes(bits);
      bits = bits & (bits - 1);
      base_ptr[base + 15] = static_cast<uint32_t>(idx) + trailingzeroes(bits);
      bits = bits & (bits - 1);
    }
    if (cnt > 16) {
      base += 16;
      do {
        base_ptr[base] = static_cast<uint32_t>(idx) + trailingzeroes(bits);
        bits = bits & (bits - 1);
        base++;
      } while (bits != 0);
    }
    base = next_base;
  }
}

#define SIMDCSV_BUFFERING 
bool find_indexes(const uint8_t * buf, size_t len, ParsedCSV & pcsv) {
  
  uint64_t prev_iter_inside_quote = 0ULL;  
#ifdef CRLF
  uint64_t prev_iter_cr_end = 0ULL; 
#endif
  size_t lenminus64 = len < 64 ? 0 : len - 64;
  size_t idx = 0;
  uint32_t *base_ptr = pcsv.indexes;
  uint32_t base = 0;
#ifdef SIMDCSV_BUFFERING
  
#define SIMDCSV_BUFFERSIZE 4 
  if(lenminus64 > 64 * SIMDCSV_BUFFERSIZE) {
    uint64_t fields[SIMDCSV_BUFFERSIZE];
    for (; idx < lenminus64 - 64 * SIMDCSV_BUFFERSIZE + 1; idx += 64 * SIMDCSV_BUFFERSIZE) {
      for(size_t b = 0; b < SIMDCSV_BUFFERSIZE; b++){
        size_t internal_idx = 64 * b + idx;
#ifndef _MSC_VER
        __builtin_prefetch(buf + internal_idx + 128);
#endif
        simd_input in = fill_input(buf+internal_idx);
        uint64_t quote_mask = find_quote_mask(in, prev_iter_inside_quote);
        uint64_t sep = cmp_mask_against_input(in, ',');
#ifdef CRLF
        uint64_t cr = cmp_mask_against_input(in, 0x0d);
        uint64_t cr_adjusted = (cr << 1) | prev_iter_cr_end;
        uint64_t lf = cmp_mask_against_input(in, 0x0a);
        uint64_t end = lf & cr_adjusted;
        prev_iter_cr_end = cr >> 63;
#else
        uint64_t end = cmp_mask_against_input(in, 0x0a);
#endif
        fields[b] = (end | sep) & ~quote_mask;
      }
      for(size_t b = 0; b < SIMDCSV_BUFFERSIZE; b++){
        size_t internal_idx = 64 * b + idx;
        flatten_bits(base_ptr, base, internal_idx, fields[b]);
      }
    }
  }
  
#endif 
  for (; idx < lenminus64; idx += 64) {
#ifndef _MSC_VER
    __builtin_prefetch(buf + idx + 128);
#endif
    simd_input in = fill_input(buf+idx);
    uint64_t quote_mask = find_quote_mask(in, prev_iter_inside_quote);
    uint64_t sep = cmp_mask_against_input(in, ',');
#ifdef CRLF
    uint64_t cr = cmp_mask_against_input(in, 0x0d);
    uint64_t cr_adjusted = (cr << 1) | prev_iter_cr_end;
    uint64_t lf = cmp_mask_against_input(in, 0x0a);
    uint64_t end = lf & cr_adjusted;
    prev_iter_cr_end = cr >> 63;
#else
    uint64_t end = cmp_mask_against_input(in, 0x0a);
#endif
    
    uint64_t field_sep = (end | sep) & ~quote_mask;
    flatten_bits(base_ptr, base, idx, field_sep);
  }
#undef SIMDCSV_BUFFERSIZE
  pcsv.n_indexes = base;
  return true;
}

int main(int argc, char * argv[]) {
  int c; 
  bool verbose = false;
  bool dump = false;
  size_t iterations = 1;
  
  while ((c = getopt(argc, argv, "vdi:s")) != -1){
    switch (c) {
    case 'v':
      verbose = true;
      break;
    case 'd':
      dump = true;
      break;
    case 'i':
      iterations = atoi(optarg);
      break;
    case 's':
      
      cerr << "problem z parametrem?" << endl;
      break;
    }
  }
  if (optind >= argc) {
    cerr << "Sposob uzycia: " << argv[0] << " <csvfile>" << endl;
    exit(1);
  }
  
  const char *filename = argv[optind];
  if (optind + 1 < argc) {
    cerr << "blad 1 " << argv[optind + 1] << endl;
  }
  
  if (verbose) {
    cout << "[info] wczytywanie " << filename << endl;
  }
  std::basic_string_view<uint8_t> p;
  try {
    p = get_corpus(filename, CSV_PADDING);
  } catch (const std::exception &e) { 
    std::cout << "Nie moze wczytac pliku " << filename << std::endl;
    return EXIT_FAILURE;
  }
  if (verbose) {
    cout << "[info] zaladowano " << filename << " (" << p.size() << " bajtow)" << endl;
  }
#ifdef __linux__
  vector<int> evts;
  evts.push_back(PERF_COUNT_HW_CPU_CYCLES);
  evts.push_back(PERF_COUNT_HW_INSTRUCTIONS);
  evts.push_back(PERF_COUNT_HW_BRANCH_MISSES);
  evts.push_back(PERF_COUNT_HW_CACHE_REFERENCES);
  evts.push_back(PERF_COUNT_HW_CACHE_MISSES);
  evts.push_back(PERF_COUNT_HW_REF_CPU_CYCLES);
#endif 
  
  ParsedCSV pcsv;
  pcsv.indexes = new (std::nothrow) uint32_t[p.size()]; 
  if(pcsv.indexes == nullptr) {
    cerr << "brak pamieci RAM" << endl;
    return EXIT_FAILURE;
  }
  
#ifdef __linux__
  TimingAccumulator ta(2, evts);
#endif 
  double total = 0; 
  for (size_t i = 0; i < iterations; i++) {
    clock_t start = clock(); 
#ifdef __linux__
{TimingPhase p1(ta, 0);
#endif 
  find_indexes(p.data(), p.size(), pcsv);
#ifdef __linux__
}{TimingPhase p2(ta, 1);} 
#endif 
total += clock() - start; 
  }
  
  if (dump) {
    for (size_t i = 0; i < pcsv.n_indexes; i++) {
      i = 0;
      cout << pcsv.indexes[i] << ": ";
      if (i != pcsv.n_indexes-1) {
        for (size_t j = pcsv.indexes[i]; j < pcsv.indexes[i+1]; j++) {
          cout << p[j];
        }
      }
      cout << "\n";
    }
  } 
  if(verbose) {
    cout << "liczba indeksow    : " << pcsv.n_indexes << endl;
    cout << "liczba bajtow na indeks : " << p.size() / double(pcsv.n_indexes) << endl;
  }
  double volume = iterations * p.size();
  double time_in_s = total / CLOCKS_PER_SEC;
  if(verbose) {
    cout << "Czas w (s)          = " << time_in_s << endl;
    cout << "Liczba iteracji      = " << volume << endl;
  } 
  cout << " GB/s: " << volume / time_in_s / (1024 * 1024 * 1024) << endl;
  if (verbose) {
    cout << "[info] koniec " << endl;
  }
  delete[] pcsv.indexes;
  aligned_free((void*)p.data());
  return EXIT_SUCCESS;
}



//////


#ifndef SIMDCSV_JSONIOUTIL_H
#define SIMDCSV_JSONIOUTIL_H

#include "common_defs.h"
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdint>

uint8_t * allocate_padded_buffer(size_t length, size_t padding);

std::basic_string_view<uint8_t>  get_corpus(const std::string& filename, size_t padding);

#endif



//////


#include "io_util.h"
#include "mem_util.h"
#include <cstring>
#include <cstdlib>

uint8_t * allocate_padded_buffer(size_t length, size_t padding) {
  
  size_t totalpaddedlength = length + padding;
  uint8_t * padded_buffer = (uint8_t *) aligned_malloc(64, totalpaddedlength);
  return padded_buffer;
}

std::basic_string_view<uint8_t> get_corpus(const std::string& filename, size_t padding) {
  std::FILE *fp = std::fopen(filename.c_str(), "rb");
  if (fp != nullptr) {
    std::fseek(fp, 0, SEEK_END);
    size_t len = std::ftell(fp);
    uint8_t * buf = allocate_padded_buffer(len, padding);
    if(buf == nullptr) {
      std::fclose(fp);
      throw  std::runtime_error("could not allocate memory");
    }
    std::rewind(fp);
    size_t readb = std::fread(buf, 1, len, fp);
    std::fclose(fp);
    if(readb != len) {
      aligned_free(buf);
      throw  std::runtime_error("could not read the data");
    }
    return std::basic_string_view<uint8_t>(buf, len+padding);
  }
  throw  std::runtime_error("could not load corpus");
}