#include <vamp/bindings/init.hh>

#ifdef VAMP_BUILD_JAX
#include <vamp/bindings/jax/jax.hh>
#endif

namespace vb = vamp::binding;

NB_MODULE(_core_ext, pymodule)
{
    vb::init_settings(pymodule);
    vb::init_environment(pymodule);
    vb::init_sphere(pymodule);
    vb::init_ur5(pymodule);
    vb::init_panda(pymodule);
    vb::init_fetch(pymodule);
    vb::init_baxter(pymodule);

#ifdef VAMP_BUILD_JAX
    vb::init_jax(pymodule);
#endif
}
