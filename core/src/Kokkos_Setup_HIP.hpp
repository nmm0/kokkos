
#if defined(KOKKOS_ENABLE_HIP)

#define KOKKOS_IMPL_HIP_CLANG_WORKAROUND

#define HIP_ENABLE_PRINTF
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#define KOKKOS_LAMBDA [=] __host__ __device__
#endif  // #if defined(KOKKOS_ENABLE_HIP)

#if defined(KOKKOS_ENABLE_HIP)

#define KOKKOS_IMPL_FORCEINLINE_FUNCTION __device__ __host__ __forceinline__
#define KOKKOS_IMPL_INLINE_FUNCTION __device__ __host__ inline
#define KOKKOS_DEFAULTED_FUNCTION __device__ __host__ inline
#define KOKKOS_INLINE_FUNCTION_DELETED __device__ __host__ inline
#define KOKKOS_IMPL_FUNCTION __device__ __host__
#define KOKKOS_IMPL_HOST_FUNCTION __host__
#define KOKKOS_IMPL_DEVICE_FUNCTION __device__
#if defined(KOKKOS_ENABLE_CXX17) || defined(KOKKOS_ENABLE_CXX20)
#define KOKKOS_CLASS_LAMBDA [ =, *this ] __host__ __device__
#endif
#endif  // #if defined( KOKKOS_ENABLE_HIP )
