#include <nanobind/nanobind.h>
#include <xla/ffi/api/c_api.h>
#include <xla/ffi/api/ffi.h>

#include <vamp/bindings/jax/jax.hh>

#include <vamp/robots/panda.hh>
#include <vamp/collision/validity.hh>
#include <vamp/planning/validate.hh>

namespace nb = nanobind;
namespace ffi = xla::ffi;

using Robot = vamp::robots::Panda;
static constexpr auto rake = 8;
using EnvironmentVector = vamp::collision::Environment<vamp::FloatVector<rake>>;

inline auto validate_motion_pairwise_impl(
    ffi::Buffer<ffi::F32> a,
    ffi::Buffer<ffi::F32> b,
    ffi::ResultBuffer<ffi::PRED> r) noexcept -> ffi::Error
{
    const auto a_d = a.dimensions();
    const auto b_d = b.dimensions();

    const auto *a_data = a.typed_data();
    const auto *b_data = b.typed_data();
    auto *r_data = r->typed_data();

    EnvironmentVector env;

    for (auto i = 0U; i < a_d[0]; ++i)
    {
        Robot::Configuration a_c(&a_data[i * 7], false);
        for (auto j = 0U; j < b_d[0]; ++j)
        {
            Robot::Configuration b_c(&b_data[j * 7], false);
            r_data[i * a_d[0] + j] = vamp::planning::validate_motion<Robot, rake, 2>(a_c, b_c, env);
        }
    }

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    validate_motion_pairwise,
    validate_motion_pairwise_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::Buffer<ffi::F32>>()   // config buffer a
        .Arg<ffi::Buffer<ffi::F32>>()   // config buffer b
        .Ret<ffi::Buffer<ffi::PRED>>()  // pairwise validity
);

template <typename T>
nb::capsule EncapsulateFfiHandler(T *fn)
{
    static_assert(
        std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
        "Encapsulated function must be and XLA FFI handler");
    return nb::capsule(reinterpret_cast<void *>(fn));
}

void vamp::binding::init_jax(nanobind::module_ &pymodule)
{
    auto submodule = pymodule.def_submodule("jax", "JAX Extensions");

    submodule.def(
        "registrations",
        []()
        {
            nb::dict registrations;
            registrations["validate_motion_pairwise"] = EncapsulateFfiHandler(validate_motion_pairwise);
            return registrations;
        });
}
