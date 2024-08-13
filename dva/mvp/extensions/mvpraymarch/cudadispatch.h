// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
// 
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#ifndef cudadispatch_h_
#define cudadispatch_h_

#include <functional>
#include <memory>
#include <type_traits>

template<typename T, typename = void>
struct get_base {
    typedef T type;
};

template<typename T>
struct get_base<T, typename std::enable_if<std::is_base_of<typename T::base, T>::value>::type> {
    typedef std::shared_ptr<typename T::base> type;
};

template<typename T> struct is_shared_ptr : std::false_type {};
template<typename T> struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

template<typename OutT, typename T>
auto convert_shptr_impl2(std::shared_ptr<T> t) {
    return *static_cast<OutT*>(t.get());
}

template<typename OutT, typename T>
auto convert_shptr_impl(T&& t, std::false_type) {
    return convert_shptr_impl2<OutT>(t);
}

template<typename OutT, typename T>
auto convert_shptr_impl(T&& t, std::true_type) {
    return std::forward<T>(t);
}

template<typename OutT, typename T>
auto convert_shptr(T&& t) {
    return convert_shptr_impl<OutT>(std::forward<T>(t), std::is_same<OutT, T>{});
}

template<typename... ArgsIn>
struct cudacall {
    struct functbase {
        virtual ~functbase() {}
        virtual void call(dim3, dim3, cudaStream_t, ArgsIn...) const = 0;
    };

    template<typename... ArgsOut>
    struct funct : public functbase {
        std::function<void(ArgsOut...)> fn;
        funct(void(*fn_)(ArgsOut...)) : fn(fn_) { }
        void call(dim3 gridsize, dim3 blocksize, cudaStream_t stream, ArgsIn... args) const {
            void (*const*kfunc)(ArgsOut...) = fn.template target<void (*)(ArgsOut...)>();
            (*kfunc)<<<gridsize, blocksize, 0, stream>>>(
                    std::forward<ArgsOut>(convert_shptr<ArgsOut>(std::forward<ArgsIn>(args)))...);
        }
    };

    std::shared_ptr<functbase> fn;

    template<typename... ArgsOut>
    cudacall(void(*fn_)(ArgsOut...)) : fn(std::make_shared<funct<ArgsOut...>>(fn_)) { }

    template<typename... ArgsTmp>
    void call(dim3 gridsize, dim3 blocksize, cudaStream_t stream, ArgsTmp&&... args) const {
        fn->call(gridsize, blocksize, stream, std::forward<ArgsIn>(args)...);
    }
};

template <typename F, typename T>
struct binder {
    F f; T t;
    template <typename... Args>
    auto operator()(Args&&... args) const
        -> decltype(f(t, std::forward<Args>(args)...)) {
        return f(t, std::forward<Args>(args)...);
    }
};

template <typename F, typename T>
binder<typename std::decay<F>::type
     , typename std::decay<T>::type> BindFirst(F&& f, T&& t) {
    return { std::forward<F>(f), std::forward<T>(t) };
}

template<typename... ArgsOut>
auto make_cudacall_(void(*fn)(ArgsOut...)) {
    return BindFirst(
            std::mem_fn(&cudacall<typename get_base<ArgsOut>::type...>::template call<typename get_base<ArgsOut>::type...>),
            cudacall<typename get_base<ArgsOut>::type...>(fn));
}

template<typename... ArgsOut>
std::function<void(dim3, dim3, cudaStream_t, typename get_base<ArgsOut>::type...)> make_cudacall(void(*fn)(ArgsOut...)) {
    return std::function<void(dim3, dim3, cudaStream_t, typename get_base<ArgsOut>::type...)>(make_cudacall_(fn));
}

#endif
