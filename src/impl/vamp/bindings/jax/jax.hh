#include <nanobind/nanobind.h>

namespace vamp::binding
{
    void init_panda_jax(nanobind::module_ &pymodule);
    void init_fetch_jax(nanobind::module_ &pymodule);
    void init_jax(nanobind::module_ &pymodule);
}
