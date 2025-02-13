#include <nanobind/nanobind.h>
#include <vamp/bindings/jax/jax.hh>
#include <vamp/robots/panda.hh>

#define VAMP_JAX_ROBOT vamp::robots::Panda
#define VAMP_JAX_ROBOT_NAME panda

#include <vamp/bindings/jax/common.hh>

#undef VAMP_JAX_ROBOT
#undef VAMP_JAX_ROBOT_NAME

void vamp::binding::init_panda_jax(nanobind::module_ &pymodule)
{
    init_robot_jax<vamp::robots::Panda>(pymodule);
}
