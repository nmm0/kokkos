#include <Kokkos_Core.hpp>
#include <impl/Kokkos_TrackDuplicates.hpp>

namespace Kokkos {
namespace Experimental {

std::map<std::string, void*> DuplicateTracker::kernel_func_list;

void DuplicateTracker::add_kernel_func(std::string name, void* func_ptr) {
  kernel_func_list[name] = func_ptr;
}

void* DuplicateTracker::get_kernel_func(std::string name) {
  return kernel_func_list[name];
}

}  // namespace Experimental
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
