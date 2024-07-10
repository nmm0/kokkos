//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <Kokkos_Core.hpp>
#include <type_traits>

#ifdef KOKKOS_ENABLE_IMPL_MDSPAN
static_assert(
    std::is_convertible_v<
        Kokkos::View<long long ****, Kokkos::LayoutRight, Kokkos::Serial>,
        Kokkos::BasicView<long long, Kokkos::dextents<size_t, 4>,
                          Kokkos::Experimental::layout_right_padded<>,
                          Kokkos::Impl::checked_reference_counted_accessor<
                              long long, Kokkos::HostSpace>>>);
static_assert(
    std::is_convertible_v<
        Kokkos::BasicView<long long, Kokkos::dextents<size_t, 4>,
                          Kokkos::Experimental::layout_right_padded<>,
                          Kokkos::Impl::checked_reference_counted_accessor<
                              long long, Kokkos::HostSpace>>,
        Kokkos::BasicView<const long long, Kokkos::dextents<size_t, 4>,
                          Kokkos::Experimental::layout_right_padded<>,
                          Kokkos::Impl::checked_reference_counted_accessor<
                              const long long, Kokkos::HostSpace>>>);
static_assert(
    std::is_convertible_v<
        Kokkos::View<long long ****, Kokkos::LayoutRight, Kokkos::Serial>,
        Kokkos::BasicView<const long long, Kokkos::dextents<size_t, 4>,
                          Kokkos::Experimental::layout_right_padded<>,
                          Kokkos::Impl::checked_reference_counted_accessor<
                              const long long, Kokkos::HostSpace>>>);
#endif
