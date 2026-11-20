// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tpu_sync/frameworks/torch/torch_host_signal_bindings.h"

#include <cstdint>

#include "ATen/core/TensorBody.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/tuple.h"
#include "tpu_sync/core/host_signal.h"
#include "tpu_sync/frameworks/torch/torch_host_signal.h"
#include "tpu_sync/frameworks/torch/torch_nanobind_utils.h"

namespace nb = nanobind;

namespace raiden {

void BindTorchHostSignal(nb::module_& m) {
  m.attr("PAGE_WORDS") = kHostSignalPageWords;
  m.attr("HEADER_WORDS") = kHostSignalHeaderWords;
  m.attr("STOPPED") = kHostSignalStopped;

  nb::class_<TorchHostSignalServer>(m, "HostSignalServer")
      .def(nb::init<>())
      // Keeps the GIL: it wraps the returned page views as Python tensors.
      .def("add", &TorchHostSignalServer::Add, nb::arg("payload"),
           nb::arg("request"), nb::arg("response"), nb::arg("staging"),
           nb::arg("retained"), nb::arg("handler"), nb::arg("user") = 0,
           nb::arg("after") = -1)
      .def("start", &TorchHostSignalServer::Start)
      // The server thread never takes the GIL, but joining it must not hold
      // the GIL against other Python threads.
      .def("stop", &TorchHostSignalServer::Stop,
           nb::call_guard<nb::gil_scoped_release>())
      .def("error", &TorchHostSignalServer::Error)
      .def_prop_ro("running", &TorchHostSignalServer::Running)
      .def("__len__", &TorchHostSignalServer::Size);
}

}  // namespace raiden
