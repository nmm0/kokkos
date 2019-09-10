
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

}  // namespace Test
