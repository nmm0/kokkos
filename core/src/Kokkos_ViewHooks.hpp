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

#ifndef KOKKOS_VIEWHOOKS_HPP
#define KOKKOS_VIEWHOOKS_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_View.hpp>

// default implementation for view holder specialization
namespace Kokkos {
namespace Impl {
template <class ViewType, class Enabled = void>
class ViewHookSpecialization {
 public:
  using view_type = ViewType;

  static inline void update_view(view_type &, const void *) {}
  static void deep_copy(unsigned char *, view_type &) {}
  static void deep_copy(view_type &, unsigned char *) {}
  static constexpr const char *m_name = "Default";
};
}  // namespace Impl
}  // namespace Kokkos

#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)

#include <functional>
#include <memory>
#include <type_traits>

namespace Kokkos {

class ViewHolderBase {
 public:
  virtual size_t span() const                = 0;
  virtual bool span_is_contiguous() const    = 0;
  virtual const void *data() const           = 0;
  virtual void *rec_ptr() const              = 0;
  virtual std::string label() const noexcept = 0;

  virtual ViewHolderBase *clone() const      = 0;
  virtual size_t data_type_size() const      = 0;
  virtual bool is_hostspace() const noexcept = 0;

  // the following are implemented in the specialization class.
  // View Holder is only a pass through implementation
  virtual void deep_copy_to_buffer(unsigned char *buff)   = 0;
  virtual void deep_copy_from_buffer(unsigned char *buff) = 0;
  virtual void update_view(const void *)                  = 0;
};

// ViewHolder derives from ViewHolderBase and it
// implement the pure virtual functions above.
template <typename View>
class ViewHolder : public ViewHolderBase {
 public:
  using view_type                = View;
  using memory_space             = typename view_type::memory_space;
  using view_hook_specialization = Impl::ViewHookSpecialization<view_type>;
  explicit ViewHolder(view_type &view) : m_view(view) {}
  size_t span() const override { return m_view.span(); }
  bool span_is_contiguous() const override {
    return m_view.span_is_contiguous();
  }
  const void *data() const override { return m_view.data(); };

  void *rec_ptr() const override {
    return (void *)m_view.impl_track().template get_record<memory_space>();
  }

  ViewHolder *clone() const override { return new ViewHolder(*this); }

  std::string label() const noexcept override { return m_view.label(); }
  size_t data_type_size() const noexcept override {
    return sizeof(typename View::value_type);
  }
  bool is_hostspace() const noexcept override {
    return std::is_same<memory_space, HostSpace>::value;
  }

  void deep_copy_to_buffer(unsigned char *buff) override {
    view_hook_specialization::deep_copy(buff, m_view);
  }

  void deep_copy_from_buffer(unsigned char *buff) override {
    view_hook_specialization::deep_copy(m_view, buff);
  }

  void update_view(const void *src_rec) override {
    view_hook_specialization::update_view(m_view, src_rec);
  }

 private:
  view_type &m_view;
};

struct ViewHooks {
  using callback_type = std::function<void(ViewHolderBase &)>;
  using copy_callback_type =
      std::function<void(ViewHolderBase &, ViewHolderBase &)>;

  template <typename F, typename ConstF>
  static void set(F &&fun, ConstF &&const_fun) {
    s_callback       = std::forward<F>(fun);
    s_const_callback = std::forward<ConstF>(const_fun);
  }

  template <typename F, typename ConstF>
  static void set_cp(F &&fun, ConstF &&const_fun) {
    s_cp_callback       = std::forward<F>(fun);
    s_cp_const_callback = std::forward<ConstF>(const_fun);
  }

  static void clear() {
    s_callback          = callback_type{};
    s_const_callback    = callback_type{};
    s_cp_callback       = copy_callback_type{};
    s_cp_const_callback = copy_callback_type{};
  }

  static bool is_set() noexcept {
    return static_cast<bool>(s_callback) ||
           static_cast<bool>(s_const_callback) ||
           static_cast<bool>(s_cp_callback) ||
           static_cast<bool>(s_cp_const_callback);
  }

  template <class DataType, class... Properties>
  static void call(const View<DataType, Properties...> &view) {
    auto holder = ViewHolder<const View<DataType, Properties...> >(view);

    do_call(holder);
  }

  template <class DataType, class... Properties>
  static typename std::enable_if<
      (!std::is_same<
           typename Kokkos::ViewTraits<DataType, Properties...>::value_type,
           typename Kokkos::ViewTraits<
               DataType, Properties...>::const_value_type>::value &&
       !std::is_same<
           typename Kokkos::ViewTraits<DataType, Properties...>::memory_space,
           Kokkos::AnonymousSpace>::value),
      void>::type
  call(View<DataType, Properties...> &dst,
       const View<DataType, Properties...> &src) {
    using non_const_view_holder = ViewHolder<View<DataType, Properties...> >;
    using const_view_holder = ViewHolder<const View<DataType, Properties...> >;
    auto src_holder         = const_view_holder(src);
    auto dst_holder         = non_const_view_holder(dst);

    do_call(dst_holder, src_holder);
  }

  template <class DataType, class... Properties>
  static typename std::enable_if<
      (std::is_same<
           typename Kokkos::ViewTraits<DataType, Properties...>::value_type,
           typename Kokkos::ViewTraits<
               DataType, Properties...>::const_value_type>::value ||
       std::is_same<
           typename Kokkos::ViewTraits<DataType, Properties...>::memory_space,
           Kokkos::AnonymousSpace>::value),
      void>::type
  call(View<DataType, Properties...> &dst,
       const View<DataType, Properties...> &src) {
    auto src_holder = ViewHolder<const View<DataType, Properties...> >(src);
    auto dst_holder = ViewHolder<const View<DataType, Properties...> >(dst);

    do_call(dst_holder, src_holder);
  }

 private:
  template <class ViewHolderType>
  static typename std::enable_if<
      !std::is_const<typename ViewHolderType::view_type::value_type>::value,
      void>::type
  do_call(ViewHolderType &view) {
    if (s_callback) s_callback(view);
  }

  template <class ViewHolderType>
  static typename std::enable_if<
      std::is_const<typename ViewHolderType::view_type::value_type>::value,
      void>::type
  do_call(ViewHolderType &view) {
    if (s_const_callback) s_const_callback(view);
  }

  template <class ViewHolderType1, class ViewHolderType2>
  static typename std::enable_if<
      !std::is_const<typename ViewHolderType1::view_type::value_type>::value,
      void>::type
  do_call(ViewHolderType1 &dst, ViewHolderType2 &src) {
    if (s_callback) s_callback(src);
    if (s_cp_callback) s_cp_callback(dst, src);
  }

  template <class ViewHolderType1, class ViewHolderType2>
  static typename std::enable_if<
      std::is_const<typename ViewHolderType1::view_type::value_type>::value,
      void>::type
  do_call(ViewHolderType1 &dst, ViewHolderType2 &src) {
    if (s_const_callback) s_const_callback(src);
    if (s_cp_const_callback) s_cp_const_callback(dst, src);
  }

  static callback_type s_callback;
  static callback_type s_const_callback;
  static copy_callback_type s_cp_callback;
  static copy_callback_type s_cp_const_callback;
};

}  // namespace Kokkos

#endif

#endif  // KOKKOS_VIEWHOOKS_HPP
