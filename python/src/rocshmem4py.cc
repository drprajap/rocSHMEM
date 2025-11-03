/*
 * Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
 * Simplified Python bindings for ROCSHMEM - Core functions only
 */

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <rocshmem/rocshmem.hpp>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace py = pybind11;
using namespace rocshmem;

#define CHECK_ROCSHMEM(expr)                                                   \
  do {                                                                         \
    int status = expr;                                                         \
    if (status != ROCSHMEM_SUCCESS) {                                          \
      std::ostringstream err_msg;                                              \
      err_msg << "ROCSHMEM error in " << __FILE__ << ":" << __LINE__;        \
      throw std::runtime_error(err_msg.str());                                 \
    }                                                                          \
  } while (0)

PYBIND11_MODULE(_rocshmem4py, m) {
  m.doc() = "Python bindings for ROCSHMEM library";

  // Initialization
  m.def("rocshmem_init", []() { rocshmem_init(); });
  m.def("rocshmem_finalize", []() { rocshmem_finalize(); });

  // PE queries
  m.def("rocshmem_my_pe", []() -> int { return rocshmem_my_pe(); });
  m.def("rocshmem_n_pes", []() -> int { return rocshmem_n_pes(); });

  // Memory management
  m.def("rocshmem_malloc", [](size_t size) -> intptr_t {
    void *ptr = rocshmem_malloc(size);
    if (ptr == nullptr) {
      throw std::runtime_error("rocshmem_malloc failed");
    }
    return (intptr_t)ptr;
  });
  m.def("rocshmem_free", [](intptr_t ptr) { rocshmem_free((void *)ptr); });
  
  m.def("rocshmem_ptr", [](intptr_t dest, int pe) -> intptr_t {
    void* remote_ptr = rocshmem_ptr((const void *)dest, pe);
    if (remote_ptr == nullptr) {
      return 0;  // Return 0 for null pointer
    }
    return (intptr_t)remote_ptr;
  }, "Get pointer to remote symmetric memory", 
     py::arg("dest"), py::arg("pe"));

  // Synchronization
  m.def("rocshmem_barrier_all", []() { rocshmem_barrier_all(); });
  m.def("rocshmem_fence", []() { rocshmem_fence(); });
  m.def("rocshmem_quiet", []() { rocshmem_quiet(); });

  // Unique ID
  m.def("rocshmem_get_uniqueid", []() -> py::bytes {
    rocshmem_uniqueid_t uid;
    CHECK_ROCSHMEM(rocshmem_get_uniqueid(&uid));
    std::string bytes((char *)&uid, sizeof(uid));
    return py::bytes(bytes);
  });

  m.def("rocshmem_init_attr", [](int rank, int nranks, py::bytes bytes) {
    rocshmem_uniqueid_t uid;
    std::string uid_str = bytes;
    if (uid_str.size() != sizeof(uid)) {
      throw std::runtime_error("rocshmem_init_attr: invalid unique ID size");
    }
    rocshmem_init_attr_t init_attr;
    memcpy(&uid, uid_str.data(), uid_str.size());
    CHECK_ROCSHMEM(rocshmem_set_attr_uniqueid_args(rank, nranks, &uid, &init_attr));
    CHECK_ROCSHMEM(rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &init_attr));
  });

  // Data transfer
  m.def("rocshmem_putmem", [](intptr_t dest, intptr_t source, size_t nelems, int pe) {
    rocshmem_putmem((void *)dest, (const void *)source, nelems, pe);
  });
  m.def("rocshmem_getmem", [](intptr_t dest, intptr_t source, size_t nelems, int pe) {
    rocshmem_getmem((void *)dest, (const void *)source, nelems, pe);
  });
  m.def("rocshmem_putmem_nbi", [](intptr_t dest, intptr_t source, size_t nelems, int pe) {
    rocshmem_putmem_nbi((void *)dest, (const void *)source, nelems, pe);
  });
  m.def("rocshmem_getmem_nbi", [](intptr_t dest, intptr_t source, size_t nelems, int pe) {
    rocshmem_getmem_nbi((void *)dest, (const void *)source, nelems, pe);
  });

  // Atomics
  m.def("rocshmem_int_atomic_fetch_add", [](intptr_t dest, int value, int pe) -> int {
    return rocshmem_int_atomic_fetch_add((int *)dest, value, pe);
  });
  m.def("rocshmem_long_atomic_fetch_add", [](intptr_t dest, long value, int pe) -> long {
    return rocshmem_long_atomic_fetch_add((long *)dest, value, pe);
  });
  m.def("rocshmem_int_atomic_compare_swap", [](intptr_t dest, int cond, int value, int pe) -> int {
    return rocshmem_int_atomic_compare_swap((int *)dest, cond, value, pe);
  });

  // Constants (using int values)
  m.attr("ROCSHMEM_SUCCESS") = py::int_(0);  // ROCSHMEM_SUCCESS value
}
