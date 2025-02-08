#include <cmath>
#include <cstdint>
#include <utility>

#include <nanobind/nanobind.h>
#include <xla/ffi/api/c_api.h>
#include <xla/ffi/api/ffi.h>

namespace nb = nanobind;
namespace ffi = xla::ffi;

auto validate_motion_pairwise(const float *a, std::size_t an, const float *b, std::size_t bn, bool *result)
    -> bool
{
    return false;
}

template <ffi::DataType T>
std::pair<int64_t, int64_t> GetDims(const ffi::Buffer<T> &buffer)
{
    auto dims = buffer.dimensions();
    if (dims.size() == 0)
    {
        return std::make_pair(0, 0);
    }
    return std::make_pair(buffer.element_count(), dims.back());
}

auto validate_motion_pairwise_impl(
    ffi::Buffer<ffi::F32> a,
    ffi::Buffer<ffi::F32> b,
    ffi::ResultBuffer<ffi::F32> r) -> ffi::Error
{
    auto a_d = a.dimensions();
    for (auto i = 0U; i < a_d.size(); ++i)
    {
        std::cout << a_d[i] << std::endl;
    }

    auto b_d = b.dimensions();
    for (auto i = 0U; i < b_d.size(); ++i)
    {
        std::cout << b_d[i] << std::endl;
    }

    // auto [totalSize, lastDim] = GetDims(a);
    // if (lastDim == 0)
    // {
    //     return ffi::Error::InvalidArgument("Validate input must be an array");
    // }
    // for (int64_t n = 0; n < totalSize; n += lastDim)
    // {
    //     ComputeValidate(eps, lastDim, &(x.typed_data()[n]), &(y->typed_data()[n]));
    // }
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    validate_motion_pairwise,
    validate_motion_pairwise_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()  // config buffer a
        .Arg<ffi::Buffer<ffi::F32>>()  // config buffer b
        .Ret<ffi::Buffer<ffi::F32>>()  // pairwise validity
);

template <typename T>
nb::capsule EncapsulateFfiHandler(T *fn)
{
    static_assert(
        std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
        "Encapsulated function must be and XLA FFI handler");
    return nb::capsule(reinterpret_cast<void *>(fn));
}

NB_MODULE(_jax_ext, pymodule)
{
    pymodule.def(
        "registrations",
        []()
        {
            nb::dict registrations;
            registrations["validate_motion_pairwise"] = EncapsulateFfiHandler(Validate);
            return registrations;
        });

    pymodule.def()
}
