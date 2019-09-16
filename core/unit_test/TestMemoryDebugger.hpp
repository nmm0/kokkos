
/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <cstdio>

#include <Kokkos_Core.hpp>

namespace Test {

template <class ExecSpace, class MemorySpace, class T>
struct TestMemoryDebugger {
  typedef Kokkos::View<T*, MemorySpace> view_type;

  size_t N = 0;

  TestMemoryDebugger(size_t n_) : N(n_) {}

  void run_test(bool run_out_of_bounds) {
    view_type A("a", N);
    view_type B("b", N);

    typename view_type::HostMirror h_A = Kokkos::create_mirror_view(A);
    typename view_type::HostMirror h_B = Kokkos::create_mirror_view(B);

    Kokkos::deep_copy(h_A, 0);
    Kokkos::deep_copy(h_B, 0);

    Kokkos::deep_copy(A, h_A);
    Kokkos::deep_copy(B, h_B);
    size_t local_N = N;

    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int i) {
          A(i) = i * 2;
          B(i) = i * 3;

          //  this section insert bad data before and after the given range.
          if (run_out_of_bounds) {
            T* tA = A.data();
            T* tB = B.data();
            tA    = (tA - 5);
            *tA   = (T)10.5;
            for (int r = 0; r < (local_N + 5); r++) {
              tB++;
            }
            *tB = (T)16.3;
          }
        });
    Kokkos::fence();
    Kokkos::deep_copy(h_A, A);
    Kokkos::deep_copy(h_B, B);

    for (int i = 0; i < N; i++) {
      KOKKOS_ASSERT(h_A(i) == i * 2);
      KOKKOS_ASSERT(h_B(i) == i * 3);
    }
    if (run_out_of_bounds) {
      bool bFail = A.verify_data();
      KOKKOS_ASSERT(bFail == false);
      bFail = B.verify_data();
      KOKKOS_ASSERT(bFail == false);
    }
  }
};

template <class ExecSpace, class MemorySpace>
struct TestNonScalarMemoryDebugger {
  struct ViewContainer {
    long part_one[3];
    long part_two[3];
    long part_three[3];
  };

  typedef Kokkos::View<ViewContainer, MemorySpace> view_type_one;
  typedef Kokkos::View<ViewContainer*, MemorySpace> view_type_multiple;

  size_t N = 0;

  TestNonScalarMemoryDebugger(size_t n_) : N(n_) {}

  void run_test(bool run_out_of_bounds) {
    view_type_one A("a");
    view_type_multiple B("b", N);

    typename view_type_one::HostMirror h_A      = Kokkos::create_mirror_view(A);
    typename view_type_multiple::HostMirror h_B = Kokkos::create_mirror_view(B);

    for (int i = 0; i < 3; i++) {
      h_A().part_one[i]   = i;
      h_A().part_two[i]   = 10 + i;
      h_A().part_three[i] = 100 + i;
    }
    for (int i = 0; i < N; i++) {
      for (int r = 0; r < 3; r++) {
        h_B(i).part_one[r]   = 0;
        h_B(i).part_two[r]   = 0;
        h_B(i).part_three[r] = 0;
      }
    }

    Kokkos::deep_copy(A, h_A);
    Kokkos::deep_copy(B, h_B);

    size_t local_N = N;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecSpace>(0, N), KOKKOS_LAMBDA(const int i) {
          for (int r = 0; r < 3; r++) {
            B(i).part_one[r]   = A().part_one[r] * i;
            B(i).part_two[r]   = A().part_two[r] * i;
            B(i).part_three[r] = A().part_three[r] * i;
          }

          //  this section insert bad data before and after the given range.
          if (run_out_of_bounds) {
            long* tA          = (long*)A.data();
            ViewContainer* tB = B.data();
            tA                = (tA - 5);
            *tA               = 10;
            for (int r = 0; r < (local_N + 5); r++) {
              tB++;
            }
            *((long*)tB) = (long)16;
          }
        });
    Kokkos::fence();
    Kokkos::deep_copy(h_B, B);

    for (int i = 0; i < N; i++) {
      for (int r = 0; r < 3; r++) {
        KOKKOS_ASSERT(h_B(i).part_one[r] == i * h_A().part_one[r]);
        KOKKOS_ASSERT(h_B(i).part_two[r] == i * h_A().part_two[r]);
        KOKKOS_ASSERT(h_B(i).part_three[r] == i * h_A().part_three[r]);
      }
    }
    if (run_out_of_bounds) {
      bool bFail = A.verify_data();
      KOKKOS_ASSERT(bFail == false);
      bFail = B.verify_data();
      KOKKOS_ASSERT(bFail == false);
    }
  }
};

TEST_F(TEST_CATEGORY, memory_debugger_good) {
  {
    TestMemoryDebugger<TEST_EXECSPACE, Kokkos::CudaSpace, int> f(100);
    f.run_test(false);
    TestMemoryDebugger<TEST_EXECSPACE, Kokkos::CudaSpace, double> f2(100);
    f2.run_test(false);
  }
}

TEST_F(TEST_CATEGORY, memory_debugger_bad) {
  {
    TestMemoryDebugger<TEST_EXECSPACE, Kokkos::CudaSpace, int> f(100);
    f.run_test(true);
    TestMemoryDebugger<TEST_EXECSPACE, Kokkos::CudaSpace, double> f2(100);
    f2.run_test(true);
  }
}

TEST_F(TEST_CATEGORY, struct_memory_debugger_good) {
  {
    TestNonScalarMemoryDebugger<TEST_EXECSPACE, Kokkos::CudaSpace> f(100);
    f.run_test(false);
  }
}

TEST_F(TEST_CATEGORY, struct_memory_debugger_bad) {
  {
    TestNonScalarMemoryDebugger<TEST_EXECSPACE, Kokkos::CudaSpace> f(100);
    f.run_test(true);
  }
}

}  // namespace Test
