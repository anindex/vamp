#include <nanobind/nanobind.h>
#include <vamp/bindings/jax/jax.hh>
#include <vamp/robots/fetch.hh>

#define VAMP_JAX_ROBOT vamp::robots::Fetch
#define VAMP_JAX_ROBOT_NAME fetch

#include <vamp/bindings/jax/common.hh>

#undef VAMP_JAX_ROBOT
#undef VAMP_JAX_ROBOT_NAME

void vamp::binding::init_fetch_jax(nanobind::module_ &pymodule)
{
    init_robot_jax<vamp::robots::Fetch>(pymodule);
}
