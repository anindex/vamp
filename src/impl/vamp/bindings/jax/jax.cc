#include <nanobind/nanobind.h>
#include <vamp/bindings/jax/jax.hh>

namespace vb = vamp::binding;

void vamp::binding::init_jax(nanobind::module_ &pymodule)
{
    auto submodule = pymodule.def_submodule("jax", "JAX Extensions");

    vb::init_panda_jax(submodule);
    vb::init_fetch_jax(submodule);
}
