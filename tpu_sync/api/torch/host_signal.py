# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Device-initiated host requests for torch_tpu compiled programs.

A compiled program posts a request (a 1-D int32 tensor computed on the
device), keeps computing, and later waits for the host's result in the same
program. A native server thread takes the request from pinned host pages,
calls a C handler that writes the result into pinned host staging, uploads it
to a retained device buffer and answers. Nothing leaves the device graph and
no Python runs per request, so the host work overlaps the device work between
``Channel.post`` and ``Channel.wait``.

Usage::

  server = host_signal.HostSignalServer()
  first = server.add_channel(request_words=n, result_words=m,
                             handler=cfunc.address, user=state_address)
  later = server.add_channel(..., predecessor=first)
  server.start()

  # Inside the compiled program, every rank in the same order:
  receipt, hidden = first.post(ids, barrier=hidden)   # before the work
  ...
  result, hidden = first.wait(receipt, words, barrier=hidden)

  server.check()  # outside the program: raises the server's first failure

The handler is a C function ``int64_t (intptr_t user, const int32_t* words,
int64_t count, uint8_t* staging, int64_t staging_bytes)`` (see
``tpu_sync/core/host_signal.h``). It runs on the server thread without the
GIL, so a numba ``cfunc`` or a C++ function fits; it returns the bytes it
wrote or a negative error. A failed request reads as zeros on the device and
every later request fails too, so the device never reads stale or partial
data.

The ops run on a mesh over every rank of the torch.distributed world, one
shard per rank, like other torch_tpu Pallas kernels: every rank must post and
wait on its channels in the same programs.

Workarounds for torch_tpu (each marked TORCHTPU-WORKAROUND below; delete it
once torch_tpu covers the case):

1. ``jax_op`` cannot declare an input's memory kind, so the callables override
   ``JaxCallable._get_jax_placeholders`` to place the pages in pinned_host.
2. torch_tpu marks only ``ao.reload`` sources as pinned host graph inputs and
   AOTAutograd's fake inputs drop the memory kind, so this module extends
   ``_backend._mark_pinned_host_inputs`` to mark the channel pages.
3. A custom op cannot mutate a pinned host input, so ``post`` returns the
   donated pages and writes them back with ``ao.offload``, which the graph
   aliases in place.
"""

import functools
import weakref
from typing import Any, cast

import torch

# Mesh axis of the channel ops: one shard per rank.
AXIS = "tpu_sync_host_signal"

_IMPL = None


def _impl():
  """Returns the host_signal submodule of the torch-backed extension."""
  global _IMPL
  if _IMPL is None:
    # pylint: disable=g-import-not-at-top
    from tpu_sync.api.torch import torch_abi
    from tpu_sync.api.torch import torch_tpu_common_loader

    torch_tpu_common_loader.load_torch_tpu_common()
    ext = torch_abi.load_extension(
        "tpu_sync.frameworks.torch",
        "_tpu_raiden_torch",
    )
    # pylint: enable=g-import-not-at-top
    _IMPL = getattr(ext, "host_signal")
  return _IMPL


def _kernels():
  # pylint: disable=g-import-not-at-top
  from tpu_sync.kernels import host_signal as kernels

  return kernels


def _pinned_host(words: int) -> torch.Tensor:
  """A zeroed ``[words]`` int32 TPU tensor in pinned host memory."""
  # pylint: disable=g-import-not-at-top,unused-import
  # Registers torch.ops.ao.offload, torch_tpu's pinned host allocation.
  from torch._functorch._activation_offloading import offload_ops  # noqa: F401
  from torch_tpu._internal import sync

  # pylint: enable=g-import-not-at-top,unused-import
  pages = torch.ops.ao.offload(torch.zeros(words, dtype=torch.int32).to("tpu"))
  sync.synchronize([pages], wait=True)
  return pages


def _device_zeros(words: int) -> torch.Tensor:
  # pylint: disable-next=g-import-not-at-top
  from torch_tpu._internal import sync

  tensor = torch.zeros(words, dtype=torch.int32, device="tpu")
  sync.synchronize([tensor], wait=True)
  return tensor


# Argument indices of the pinned host pages of each op. They must be graph
# inputs.
_HOST_ARGS = {"post": (3, 4), "wait": (1,)}


@functools.cache
def _ops() -> Any:
  """Registers ``torch.ops.tpu_sync_host_signal.*`` once torch.distributed is up."""
  # pylint: disable=g-import-not-at-top
  import jax
  from jax import lax
  from jax.sharding import NamedSharding
  from jax.sharding import PartitionSpec as P
  from torch_tpu._internal.pallas import get_pallas_mesh
  from torch_tpu._internal.pallas.pallas import JaxCallable

  # pylint: enable=g-import-not-at-top
  kernels = _kernels()
  mesh = get_pallas_mesh(axis_names=(AXIS,))
  shard = P(AXIS)

  # TORCHTPU-WORKAROUND 1: trace placeholders with their memory kinds.
  class HostCallable(JaxCallable):
    placements: tuple[Any, ...] = ()

    def _get_jax_placeholders(self, norm_args, norm_kwargs=None):
      values, keywords = super()._get_jax_placeholders(norm_args, norm_kwargs)
      return [
          jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=s)
          for v, s in zip(values, self.placements, strict=True)
      ], keywords

  def make_callable(name, body, num_args, num_outputs, host, donated=()):
    in_specs = (shard,) * num_args
    out_specs = (shard,) * num_outputs
    placements = tuple(
        NamedSharding(
            mesh.abstract_mesh,
            shard,
            memory_kind="pinned_host" if i in host else "device",
        )
        for i in range(num_args)
    )
    mapped = jax.shard_map(
        body, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
        check_vma=False,
    )
    call = HostCallable(
        name=name,
        trace_key=name,
        jit_fn=jax.jit(
            mapped, in_shardings=placements, donate_argnums=donated
        ),
        mesh=mesh,
        input_partition_specs=in_specs,
        output_partition_specs=out_specs,
        donate_argnums=list(donated),
    )
    call.placements = placements
    return call

  def post_body(words, step, after, payload_page, request_page, barrier):
    generation = kernels.next_generation(step)
    receipt, payload_page, request_page = kernels.post(
        words, generation, after, payload_page, request_page
    )
    # The request is issued before whatever consumes `barrier`.
    receipt, barrier = lax.optimization_barrier((receipt, barrier))
    return (
        receipt,
        jax.device_put(payload_page, jax.memory.Space.Host),
        jax.device_put(request_page, jax.memory.Space.Host),
        barrier,
    )

  post = make_callable(
      "tpu_sync_host_signal_post",
      post_body,
      num_args=6,
      num_outputs=4,
      host=_HOST_ARGS["post"],
      donated=_HOST_ARGS["post"],
  )

  @functools.cache
  def wait_callable(words: int):
    def wait_body(receipt, response_page, retained, barrier):
      # The wait starts only once `barrier` is computed ...
      receipt, barrier = lax.optimization_barrier((receipt, barrier))
      result = kernels.wait(receipt, response_page, retained, words)
      # ... and whatever consumes `barrier` afterwards follows it.
      return lax.optimization_barrier((result, barrier))

    return make_callable(
        f"tpu_sync_host_signal_wait_{words}",
        wait_body,
        num_args=4,
        num_outputs=2,
        host=_HOST_ARGS["wait"],
    )

  @torch.library.custom_op("tpu_sync_host_signal::post", mutates_args=())
  def post_op(
      words: torch.Tensor,
      step: torch.Tensor,
      after: torch.Tensor,
      payload_page: torch.Tensor,
      request_page: torch.Tensor,
      barrier: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = post(words, step, after, payload_page, request_page, barrier)
    return cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], outputs
    )

  @post_op.register_fake
  def _(words, step, after, payload_page, request_page, barrier):
    del words, step, after
    return (
        torch.empty_like(request_page),
        torch.empty_like(payload_page),
        torch.empty_like(request_page),
        torch.empty_like(barrier),
    )

  @torch.library.custom_op("tpu_sync_host_signal::wait", mutates_args=())
  def wait_op(
      receipt: torch.Tensor,
      response_page: torch.Tensor,
      retained: torch.Tensor,
      barrier: torch.Tensor,
      words: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = wait_callable(int(words))(
        receipt, response_page, retained, barrier
    )
    return cast(tuple[torch.Tensor, torch.Tensor], outputs)

  @wait_op.register_fake
  def _(receipt, response_page, retained, barrier, words):
    del receipt, response_page
    return retained.new_empty((words,)), torch.empty_like(barrier)

  _mark_host_pages({
      torch.ops.tpu_sync_host_signal.post.default: _HOST_ARGS["post"],
      torch.ops.tpu_sync_host_signal.wait.default: _HOST_ARGS["wait"],
  })
  return torch.ops.tpu_sync_host_signal


def _mark_host_pages(host_args: dict[Any, tuple[int, ...]]) -> None:
  """TORCHTPU-WORKAROUND 2: keep the pages pinned_host graph inputs."""
  # pylint: disable-next=g-import-not-at-top
  from torch_tpu._internal.compile import _backend

  # pylint: disable=protected-access
  mark = _backend._mark_pinned_host_inputs

  def mark_with_channel_pages(graph_module, example_inputs):
    mark(graph_module, example_inputs)
    placeholders = [n for n in graph_module.graph.nodes if n.op == "placeholder"]
    for node in graph_module.graph.nodes:
      for index in host_args.get(node.target, ()):
        page = node.args[index]
        if page not in placeholders:
          raise ValueError(
              "tpu_sync host signal pages must be graph inputs; pass the"
              " channel's tensors to the op directly"
          )
        example_inputs[placeholders.index(page)]._is_pinned_host = True

  _backend._mark_pinned_host_inputs = mark_with_channel_pages
  # pylint: enable=protected-access


def _handler_address(handler: Any) -> int:
  if isinstance(handler, int):
    return handler
  address = getattr(handler, "address", None)  # numba cfunc
  if isinstance(address, int):
    return address
  raise TypeError(
      "handler must be a C function address or have an integer .address"
  )


class Channel:
  """One request channel: pages, upload target and generation counter.

  Created by ``HostSignalServer.add_channel``. Its tensors keep their
  addresses until the server stops; programs pass them as graph inputs.
  """

  def __init__(
      self,
      server: "HostSignalServer",
      *,
      request_words: int,
      result_words: int,
      handler: Any,
      user: int,
      predecessor: "Channel | None",
  ):
    kernels = _kernels()
    if request_words <= 0 or result_words <= 0:
      raise ValueError("request_words and result_words must be positive")
    if predecessor is not None and predecessor.server is not server:
      raise ValueError("a predecessor must belong to the same server")
    _ops()
    self.server = server
    self.predecessor = predecessor
    self.request_words = request_words
    self.result_words = kernels.round_pages(result_words)
    payload = _pinned_host(kernels.payload_words(request_words))
    request = _pinned_host(kernels.PAGE_WORDS)
    self.response = _pinned_host(kernels.PAGE_WORDS)
    staging = _pinned_host(self.result_words)
    self.retained = _device_zeros(self.result_words)
    # The last generation posted; `post` advances it in the program.
    self.step = _device_zeros(1)
    self._no_predecessor = _device_zeros(1)
    self._handler = handler  # Keeps a cfunc (and its code) alive.
    self.index, self.payload, self.request = server._native.add(  # pylint: disable=protected-access
        payload,
        request,
        self.response,
        staging,
        self.retained,
        _handler_address(handler),
        user,
        predecessor.index if predecessor is not None else -1,
    )
    # Programs donate the views above; these keep the allocations.
    self._owners = (payload, request, staging)

  def post(
      self, words: torch.Tensor, *, barrier: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Posts ``words`` (1-D int32, at most ``request_words``).

    Post the predecessor's request first in the same program: its updated
    ``step`` is the generation the host serves this request after.

    Returns:
      The receipt for ``wait``, and ``barrier`` ordered after the request.
    """
    if words.dtype != torch.int32 or words.dim() != 1:
      raise ValueError("words must be a 1-D int32 tensor")
    after = (
        self.predecessor.step
        if self.predecessor is not None
        else self._no_predecessor
    )
    # The ops were registered when the channel was created; reach them
    # directly so Dynamo traces no cached helper.
    receipt, payload, request, barrier = torch.ops.tpu_sync_host_signal.post(
        words, self.step, after, self.payload, self.request, barrier
    )
    # TORCHTPU-WORKAROUND 3: write the donated pages back so the graph
    # aliases them in place.
    self.payload.copy_(torch.ops.ao.offload(payload))
    self.request.copy_(torch.ops.ao.offload(request))
    self.step.copy_(receipt[:1])
    return receipt, barrier

  def wait(
      self, receipt: torch.Tensor, words: int, *, barrier: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Waits for ``receipt`` and returns the first ``words`` result words.

    Returns:
      A fresh ``[words]`` int32 tensor (zeros if the request failed), and
      ``barrier`` ordered after the wait.
    """
    # A symbolic size (a traced token axis) is checked by the kernel at
    # lowering; comparing it here would only add a guard.
    if isinstance(words, int) and not 0 < words <= self.result_words:
      raise ValueError(
          f"cannot read {words} words from a {self.result_words}-word result"
      )
    return torch.ops.tpu_sync_host_signal.wait(
        receipt, self.response, self.retained, barrier, words
    )


class HostSignalServer:
  """One native thread serving an ordered set of channels.

  Add every channel, then ``start``. Lower-index channels are served first
  when several requests are ready; a channel with a predecessor serves a
  request only after the predecessor answered the generation it names.
  """

  def __init__(self):
    self._native = _impl().HostSignalServer()
    self._channels: list[Channel] = []
    # Answers STOPPED on every page before the pages can be released.
    self._finalizer = weakref.finalize(self, self._native.stop)

  def add_channel(
      self,
      *,
      request_words: int,
      result_words: int,
      handler: Any,
      user: int = 0,
      predecessor: Channel | None = None,
  ) -> Channel:
    """Adds a channel; see ``Channel``.

    Args:
      request_words: The most words one request carries.
      result_words: The most result words one request produces.
      handler: The C handler's address, or an object with an integer
        ``address`` (e.g. a numba cfunc), which the channel keeps alive.
      user: Passed to the handler unchanged.
      predecessor: An earlier channel of this server that this one's
        requests are served after.

    Returns:
      The channel.
    """
    channel = Channel(
        self,
        request_words=request_words,
        result_words=result_words,
        handler=handler,
        user=user,
        predecessor=predecessor,
    )
    self._channels.append(channel)
    return channel

  def start(self) -> None:
    self._native.start()

  def stop(self) -> None:
    """Stops serving and answers STOPPED; no program may still be waiting."""
    self._native.stop()

  def check(self) -> None:
    """Raises the server's first failure, if any."""
    if error := self._native.error():
      raise RuntimeError(f"tpu_sync host signal server failed: {error}")

  @property
  def running(self) -> bool:
    return self._native.running

  @property
  def channels(self) -> tuple[Channel, ...]:
    return tuple(self._channels)
