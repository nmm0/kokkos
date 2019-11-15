/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
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

/** @file Kokkos_MemorySpace.hpp
 *
 *  Operations common to memory space instances, or at least default
 *  implementations thereof.
 */

#ifndef KOKKOS_IMPL_MEMORYSPACE_HPP
#define KOKKOS_IMPL_MEMORYSPACE_HPP

#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>
#include <impl/Kokkos_Error.hpp>

#include <string>
#include <iostream>
#include <sstream>
#include <cstring>

#if defined(KOKKOS_ENABLE_PROFILING)
#include <impl/Kokkos_Profiling_Interface.hpp>
#endif

namespace Kokkos {
namespace Impl {

template <class MemorySpace>
SharedAllocationHeader *checked_allocation_with_header(MemorySpace const &space,
                                                       std::string const &label,
                                                       size_t alloc_size) {
  try {
    return reinterpret_cast<SharedAllocationHeader *>(
        space.allocate(alloc_size + sizeof(SharedAllocationHeader)));
  } catch (Kokkos::Experimental::RawMemoryAllocationFailure const &failure) {
    auto generate_failure_message = [&](std::ostream &o) {
      o << "Kokkos failed to allocate memory for label \"" << label
        << "\".  Allocation using MemorySpace named \"" << space.name()
        << "\" failed with the following error:  ";
      failure.print_error_message(o);
      if (failure.failure_mode() ==
          Kokkos::Experimental::RawMemoryAllocationFailure::FailureMode::
              AllocationNotAligned) {
        // TODO: delete the misaligned memory?
        o << "Warning: Allocation failed due to misalignment; memory may "
             "be leaked."
          << std::endl;
      }
      o.flush();
    };
    try {
      std::ostringstream sstr;
      generate_failure_message(sstr);
      Kokkos::Impl::throw_runtime_exception(sstr.str());
    } catch (std::bad_alloc const &) {
      // Probably failed to allocate the string because we're so close to out
      // of memory. Try printing to std::cerr instead
      try {
        generate_failure_message(std::cerr);
      } catch (std::bad_alloc const &) {
        // oh well, we tried...
      }
      Kokkos::Impl::throw_runtime_exception(
          "Kokkos encountered an allocation failure, then another allocation "
          "failure while trying to create the error message.");
    }
  }
  return nullptr;  // unreachable
}

template <class MemorySpace>
class SharedAllocationRecord<MemorySpace, void>
    : public SharedAllocationRecord<void, void> {
 private:
  friend MemorySpace;

  typedef SharedAllocationRecord<void, void> RecordBase;

  SharedAllocationRecord(const SharedAllocationRecord &) = delete;
  SharedAllocationRecord &operator=(const SharedAllocationRecord &) = delete;

  inline static void deallocate(RecordBase *arg_rec) {
    delete static_cast<SharedAllocationRecord *>(arg_rec);
  }

// todo: this probably won't compile...use some sort of static map to get the
// root records.
#ifdef KOKKOS_DEBUG
  /**\brief  Root record for tracked allocations from this HostSpace instance */
  static inline RecordBase s_root_record;
#endif

  const MemorySpace m_space;

 protected:
  inline ~SharedAllocationRecord() {
#if defined(KOKKOS_ENABLE_PROFILING)
    if (Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::deallocateData(
          Kokkos::Profiling::SpaceHandle(MemorySpace::name()),
          RecordBase::m_alloc_ptr->m_label, data(), size());
    }
#endif

    m_space.deallocate(RecordBase::m_alloc_ptr, RecordBase::m_alloc_size);
  }
  SharedAllocationRecord() = default;

  inline SharedAllocationRecord(
      const MemorySpace &arg_space, const std::string &arg_label,
      const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &deallocate)
      : RecordBase(
#ifdef KOKKOS_DEBUG
            // todo: this probably won't compile...use some sort of static map
            // to get the root records.
            &SharedAllocationRecord<MemorySpace, void>::s_root_record,
#endif
            Impl::checked_allocation_with_header(arg_space, arg_label,
                                                 arg_alloc_size),
            sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc),
        m_space(arg_space) {
#if defined(KOKKOS_ENABLE_PROFILING)
    if (Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::allocateData(
          Kokkos::Profiling::SpaceHandle(arg_space.name()), arg_label, data(),
          arg_alloc_size);
    }
#endif
    // Fill in the Header information
    RecordBase::m_alloc_ptr->m_record =
        static_cast<SharedAllocationRecord<void, void> *>(this);

    strncpy(RecordBase::m_alloc_ptr->m_label, arg_label.c_str(),
            SharedAllocationHeader::maximum_label_length);
    // Set last element zero, in case c_str is too long
    RecordBase::m_alloc_ptr
        ->m_label[SharedAllocationHeader::maximum_label_length - 1] = (char)0;
  }

 public:
  inline std::string get_label() const {
    return std::string(RecordBase::head()->m_label);
  }

  KOKKOS_INLINE_FUNCTION static SharedAllocationRecord *allocate(
      const Kokkos::HostSpace &arg_space, const std::string &arg_label,
      const size_t arg_alloc_size) {
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
    return new SharedAllocationRecord(arg_space, arg_label, arg_alloc_size);
#else
    return (SharedAllocationRecord *)0;
#endif
  }

  /**\brief  Allocate tracked memory in the space */
  inline static void *allocate_tracked(const MemorySpace &arg_space,
                                       const std::string &arg_label,
                                       const size_t arg_alloc_size) {
    if (!arg_alloc_size) return (void *)nullptr;

    SharedAllocationRecord *const r =
        allocate(arg_space, arg_label, arg_alloc_size);

    RecordBase::increment(r);

    return r->data();
  }

  /**\brief  Reallocate tracked memory in the space */
  inline static void *reallocate_tracked(void *const arg_alloc_ptr,
                                         const size_t arg_alloc_size) {
    SharedAllocationRecord *const r_old = get_record(arg_alloc_ptr);
    SharedAllocationRecord *const r_new =
        allocate(r_old->m_space, r_old->get_label(), arg_alloc_size);

    Kokkos::Impl::DeepCopy<MemorySpace, MemorySpace>(
        r_new->data(), r_old->data(), std::min(r_old->size(), r_new->size()));

    RecordBase::increment(r_new);
    RecordBase::decrement(r_old);

    return r_new->data();
  }

  /**\brief  Deallocate tracked memory in the space */
  inline static void deallocate_tracked(void *const arg_alloc_ptr) {
    if (arg_alloc_ptr != 0) {
      SharedAllocationRecord *const r = get_record(arg_alloc_ptr);

      RecordBase::decrement(r);
    }
  }

  inline static SharedAllocationRecord *get_record(void *alloc_ptr) {
    typedef SharedAllocationHeader Header;
    typedef SharedAllocationRecord<MemorySpace, void> RecordHost;

    SharedAllocationHeader const *const head =
        alloc_ptr ? Header::get_header(alloc_ptr) : (SharedAllocationHeader *)0;
    RecordHost *const record =
        head ? static_cast<RecordHost *>(head->m_record) : (RecordHost *)0;

    if (!alloc_ptr || record->m_alloc_ptr != head) {
      Kokkos::Impl::throw_runtime_exception(std::string(
          "Kokkos::Impl::SharedAllocationRecord< Kokkos::HostSpace , "
          "void >::get_record ERROR"));
    }

    return record;
  }

// Iterate records to print orphaned memory ...
#ifdef KOKKOS_DEBUG
  inline static void print_records(std::ostream &s, const MemorySpace &,
                                   bool detail) {
    RecordBase::print_host_accessible_records(s, "HostSpace", &s_root_record,
                                              detail);
  }
#else
  inline static void print_records(std::ostream &, const MemorySpace &, bool) {
    throw_runtime_exception(
        "SharedAllocationRecord<HostSpace>::print_records only works with "
        "KOKKOS_DEBUG enabled");
  }
#endif
};

namespace {

/* Taking the address of this function so make sure it is unique */
template <class MemorySpace, class DestroyFunctor>
void deallocate(SharedAllocationRecord<void, void> *record_ptr) {
  typedef SharedAllocationRecord<MemorySpace, void> base_type;
  typedef SharedAllocationRecord<MemorySpace, DestroyFunctor> this_type;

  this_type *const ptr =
      static_cast<this_type *>(static_cast<base_type *>(record_ptr));

  ptr->m_destroy.destroy_shared_allocation();

  delete ptr;
}

}  // namespace

/*
 *  Memory space specialization of SharedAllocationRecord< Space , void >
 * requires :
 *
 *  SharedAllocationRecord< Space , void > : public SharedAllocationRecord< void
 * , void >
 *  {
 *    // delete allocated user memory via static_cast to this type.
 *    static void deallocate( const SharedAllocationRecord<void,void> * );
 *    Space m_space ;
 *  }
 */
template <class MemorySpace, class DestroyFunctor>
class SharedAllocationRecord
    : public SharedAllocationRecord<MemorySpace, void> {
 private:
  SharedAllocationRecord(const MemorySpace &arg_space,
                         const std::string &arg_label, const size_t arg_alloc)
      /*  Allocate user memory as [ SharedAllocationHeader , user_memory ] */
      : SharedAllocationRecord<MemorySpace, void>(
            arg_space, arg_label, arg_alloc,
            &Kokkos::Impl::deallocate<MemorySpace, DestroyFunctor>),
        m_destroy() {}

  SharedAllocationRecord()                               = delete;
  SharedAllocationRecord(const SharedAllocationRecord &) = delete;
  SharedAllocationRecord &operator=(const SharedAllocationRecord &) = delete;

 public:
  DestroyFunctor m_destroy;

  // Allocate with a zero use count.  Incrementing the use count from zero to
  // one inserts the record into the tracking list.  Decrementing the count from
  // one to zero removes from the trakcing list and deallocates.
  KOKKOS_INLINE_FUNCTION static SharedAllocationRecord *allocate(
      const MemorySpace &arg_space, const std::string &arg_label,
      const size_t arg_alloc) {
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
    return new SharedAllocationRecord(arg_space, arg_label, arg_alloc);
#else
    return (SharedAllocationRecord *)0;
#endif
  }
};

}  // end namespace Impl
}  // end namespace Kokkos

#endif  // KOKKOS_IMPL_MEMORYSPACE_HPP
