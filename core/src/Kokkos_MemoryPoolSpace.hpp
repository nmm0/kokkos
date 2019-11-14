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

#ifndef KOKKOS_MEMORYPOOL_SPACE_HPP
#define KOKKOS_MEMORYPOOL_SPACE_HPP

#include <cstring>
#include <string>
#include <iosfwd>
#include <typeinfo>

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <Kokkos_MemoryPool.hpp>
#include <impl/Kokkos_MemorySpace.hpp>

#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>

/*--------------------------------------------------------------------------*/

namespace Kokkos {

namespace Impl {


}  // namespace Impl

}  // namespace Kokkos

namespace Kokkos {

/// \class MemoryPoolSpace
/// \brief Memory management for host memory.
///
/// MemoryPoolSpace is a memory space that governs host memory.  "Host"
/// memory means the usual CPU-accessible memory.
template <class RootDevice>
class MemoryPoolSpace {
 protected:
  enum { MaxCapacity = 16000 };
 public:
  //! Tag this class as a kokkos memory space
  using root_memory_space = typename RootDevice::memory_space;
  using memory_space = MemoryPoolSpace;
  using execution_space = typename RootDevice::execution_space;
  using size_type = size_t;

  //! This memory space preferred device_type
  using device_type = Kokkos::Device<execution_space, root_memory_space>;
  using memory_pool = Kokkos::MemoryPool<device_type>;

  /**\brief  Default memory space instance */
  MemoryPoolSpace() : mem_pool( root_memory_space(), MaxCapacity ) {
  }
  MemoryPoolSpace(MemoryPoolSpace&& rhs)      = default;
  MemoryPoolSpace(const MemoryPoolSpace& rhs) = default;
  MemoryPoolSpace& operator=(MemoryPoolSpace&&) = default;
  MemoryPoolSpace& operator=(const MemoryPoolSpace&) = default;
  ~MemoryPoolSpace()                           = default;

  // have to provide a device to instantiate...
  explicit MemoryPoolSpace(const memory_pool & mp_) : mem_pool(mp_) {
  }


  /**\brief  Allocate untracked memory in the space */
  void* allocate(const size_t arg_alloc_size) const {
     void *ptr = nullptr;

     if (arg_alloc_size) {
        ptr = mem_pool.allocate(arg_alloc_size); /* Assume that the memory pool will handle failures */
     }

     return ptr;
  }


  /**\brief  Deallocate untracked memory in the space */
  void deallocate(void* const arg_alloc_ptr, const size_t arg_alloc_size) const {
     if (arg_alloc_ptr) {
        mem_pool.deallocate(arg_alloc_ptr, arg_alloc_size); 
     }
  }


  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; }

 private:
  const memory_pool mem_pool; /* Kokkos::MemoryPool is ref counted so we can just keep a copy here */
  static constexpr const char* m_name = "MemoryPoolSpace";
  friend class Kokkos::Impl::SharedAllocationRecord<MemoryPoolSpace, void>;
};

}  // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

// Need an alternative for this
/*static_assert(Kokkos::Impl::MemorySpaceAccess<Kokkos::MemoryPoolSpace,
                                              Kokkos::MemoryPoolSpace>::assignable,
              "");*/

}  // namespace Impl

}  // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

template <>
class SharedAllocationRecord<Kokkos::MemoryPoolSpace< Kokkos::Device< Kokkos::Serial, 
                                                                      Kokkos::HostSpace
                                                                    >
                                                    >,
                             void
                            > 
    : public SharedAllocationRecord<void, void> {
 private:
  using memory_space = Kokkos::MemoryPoolSpace< Kokkos::Device< Kokkos::Serial,
                                                                Kokkos::HostSpace
                                                              >
                                              >;
  friend memory_space;

  typedef SharedAllocationRecord<void, void> RecordBase;

  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

  static void deallocate(RecordBase* arg_rec) {
     delete static_cast<SharedAllocationRecord *>(arg_rec);
  }


#ifdef KOKKOS_DEBUG
  /**\brief  Root record for tracked allocations from this MemoryPoolSpace instance */
  static RecordBase s_root_record;
#endif

  const memory_space m_space;

 protected:
  inline 
  ~SharedAllocationRecord() {
#if defined(KOKKOS_ENABLE_PROFILING)
     if (Kokkos::Profiling::profileLibraryLoaded()) {
       Kokkos::Profiling::deallocateData(
           Kokkos::Profiling::SpaceHandle(Kokkos::HostSpace::name()),
           RecordBase::m_alloc_ptr->m_label, data(), size());
     }
#endif

     m_space.deallocate(SharedAllocationRecord<void, void>::m_alloc_ptr,
                        SharedAllocationRecord<void, void>::m_alloc_size);
  }
  SharedAllocationRecord() = default;

  inline
  SharedAllocationRecord(
      const memory_space& arg_space, const std::string& arg_label,
      const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &deallocate)
       : SharedAllocationRecord<void, void>(
#ifdef KOKKOS_DEBUG
             &SharedAllocationRecord<memory_space, void>::s_root_record,
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

  KOKKOS_INLINE_FUNCTION static SharedAllocationRecord* allocate(
      const memory_space& arg_space, const std::string& arg_label,
      const size_t arg_alloc_size) {
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
    return new SharedAllocationRecord(arg_space, arg_label, arg_alloc_size);
#else
    return (SharedAllocationRecord*)0;
#endif
  }

  /**\brief  Allocate tracked memory in the space */
  inline
  static void* allocate_tracked(const memory_space& arg_space,
                                const std::string& arg_label,
                                const size_t arg_alloc_size) {
     if (!arg_alloc_size) return (void *)nullptr;

     SharedAllocationRecord *const r =
         allocate(arg_space, arg_label, arg_alloc_size);

     RecordBase::increment(r);

     return r->data();
  }

  /**\brief  Reallocate tracked memory in the space */
  inline
  static void* reallocate_tracked(void* const arg_alloc_ptr,
                                  const size_t arg_alloc_size) {
     SharedAllocationRecord *const r_old = get_record(arg_alloc_ptr);
     SharedAllocationRecord *const r_new =
         allocate(r_old->m_space, r_old->get_label(), arg_alloc_size);

     Kokkos::Impl::DeepCopy<typename memory_space::root_memory_space, typename memory_space::root_memory_space>(
         r_new->data(), r_old->data(), std::min(r_old->size(), r_new->size()));

     RecordBase::increment(r_new);
     RecordBase::decrement(r_old);

     return r_new->data();
  }

  /**\brief  Deallocate tracked memory in the space */
  inline
  static void deallocate_tracked(void* const arg_alloc_ptr) {
     if (arg_alloc_ptr != 0) {
       SharedAllocationRecord *const r = get_record(arg_alloc_ptr);

       RecordBase::decrement(r);
     }
  }

  inline
  static SharedAllocationRecord* get_record(void* arg_alloc_ptr) {
     typedef SharedAllocationHeader Header;
     typedef SharedAllocationRecord<memory_space, void> RecordHost;

     SharedAllocationHeader const *const head =
         arg_alloc_ptr ? Header::get_header(arg_alloc_ptr) : (SharedAllocationHeader *)0;

     RecordHost *const record =
         head ? static_cast<RecordHost *>(head->m_record) : (RecordHost *)0;

     if (!arg_alloc_ptr || !record || record->m_alloc_ptr != head) {
       Kokkos::Impl::throw_runtime_exception(
           std::string("Kokkos::Impl::SharedAllocationRecord< Kokkos::MemoryPoolSpace , "
                       "void >::get_record ERROR"));
     }

     return record;
  }

  inline
  static void print_records(std::ostream&, const memory_space&,
                            bool detail = false) {
  }
};

}  // namespace Impl

}  // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

template <class ExecutionSpace>
struct DeepCopy<Kokkos::MemoryPoolSpace< Kokkos::Device<ExecutionSpace, Kokkos::HostSpace> >, Kokkos::HostSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    using memory_pool_space = Kokkos::MemoryPoolSpace< Kokkos::Device<ExecutionSpace, Kokkos::HostSpace> >;
    using source_space = Kokkos::HostSpace;
    using dest_space = typename memory_pool_space::root_memory_space;
    Kokkos::Impl::DeepCopy<dest_space,source_space,ExecutionSpace>(dst, src, n);
  }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    using memory_pool_space = Kokkos::MemoryPoolSpace< Kokkos::Device<ExecutionSpace, Kokkos::HostSpace> >;
    using source_space = Kokkos::HostSpace;
    using dest_space = typename memory_pool_space::root_memory_space;
    Kokkos::Impl::DeepCopy<dest_space,source_space,ExecutionSpace>(exec,dst, src, n);
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::HostSpace,Kokkos::MemoryPoolSpace< Kokkos::Device<ExecutionSpace, Kokkos::HostSpace> >, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    using memory_pool_space = Kokkos::MemoryPoolSpace< Kokkos::Device<ExecutionSpace, Kokkos::HostSpace> >;
    using source_space = typename memory_pool_space::root_memory_space; 
    using dest_space = Kokkos::HostSpace;
    Kokkos::Impl::DeepCopy<dest_space,source_space,ExecutionSpace>(dst, src, n);
  }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    using memory_pool_space = Kokkos::MemoryPoolSpace< Kokkos::Device<ExecutionSpace, Kokkos::HostSpace> >;
    using source_space = typename memory_pool_space::root_memory_space; 
    using dest_space = Kokkos::HostSpace;
    Kokkos::Impl::DeepCopy<dest_space,source_space,ExecutionSpace>(exec,dst, src, n);
  }
};

}  // namespace Impl

}  // namespace Kokkos

#endif  // #define KOKKOS_MEMORYPOOL_SPACE_HPP
