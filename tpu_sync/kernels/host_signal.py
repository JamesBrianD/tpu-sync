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

"""Device side of the host signal channel (tpu_sync/core/host_signal.h).

A compiled program posts a request with ``post`` and collects the host's
result with ``wait``:

* ``post`` writes ``[count, after, words...]`` to the pinned host payload page
  and only then the generation to the request page, so the host never sees a
  generation before its payload. It returns the generation as an HBM receipt.
* ``wait`` polls the response page until the host answers the receipt's
  generation (itself on success, its negation on failure, ``STOPPED`` once the
  server stopped), then copies the device buffer the host uploaded into a
  fresh output. Returning the externally written buffer itself would let XLA
  read or relayout it before the upload lands. A failed or stopped request
  reads as zeros, never as stale data.

Pure JAX, so torch_tpu and JAX programs share it. The callers place the pages
in pinned host memory and donate the payload and request pages.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import mosaic
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc

# Must match kHostSignalPageWords / kHostSignalHeaderWords / kHostSignalStopped.
PAGE_WORDS = 1024
HEADER_WORDS = 2
STOPPED = -(2**31)
_INT32_MAX = 2**31 - 1

_SC_PARAMS = pltpu.CompilerParams(opt_level=mosaic.OptLevel.O3)


def round_pages(words: int) -> int:
  """``words`` rounded up to whole pages."""
  return -(-words // PAGE_WORDS) * PAGE_WORDS


def payload_words(capacity: int) -> int:
  """Payload page words for requests of up to ``capacity`` words."""
  return round_pages(capacity + HEADER_WORDS)


def next_generation(step: jax.Array) -> jax.Array:
  """The generation after ``step[0]``: positive and never ``step[0]`` again."""
  return jnp.where(step[0] == _INT32_MAX, 1, step[0] + 1).astype(jnp.int32)


def _scalar_mesh() -> plsc.ScalarSubcoreMesh:
  return plsc.ScalarSubcoreMesh(axis_name="tpu_sync_host_signal", num_cores=1)


def post(
    words: jax.Array,
    generation: jax.Array,
    after: jax.Array,
    payload_page: jax.Array,
    request_page: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Posts ``words`` (1-D int32) as request ``generation``.

  Args:
    words: The request's words.
    generation: Scalar int32 from ``next_generation``.
    after: ``[1]`` int32, a predecessor generation the host must answer first,
      or 0.
    payload_page: The channel's payload page (pinned host, donated).
    request_page: The channel's request page (pinned host, donated).

  Returns:
    The HBM receipt page (every word the generation) and the updated pages.
  """
  if words.ndim != 1:
    raise ValueError(f"words must be 1-D, got shape {words.shape}")
  payload = jnp.concatenate((
      jnp.array([words.shape[0]], jnp.int32),
      after.reshape(1).astype(jnp.int32),
      words.astype(jnp.int32),
  ))
  payload = jnp.pad(payload, (0, round_pages(payload.size) - payload.size))
  if payload.size > payload_page.size:
    raise ValueError(
        f"request of {words.shape[0]} words exceeds the channel capacity of "
        f"{payload_page.size - HEADER_WORDS}"
    )
  page = jnp.broadcast_to(generation.astype(jnp.int32), (PAGE_WORDS,))
  payload_ref = jax.new_ref(payload)
  generation_ref = jax.new_ref(page)
  target_ref = jax.new_ref(payload_page, memory_space=pl.HOST)
  request_ref = jax.new_ref(request_page, memory_space=pl.HOST)

  @pl.kernel(
      out_type=pltpu.HBM((PAGE_WORDS,), jnp.int32),
      mesh=_scalar_mesh(),
      compiler_params=_SC_PARAMS,
      scratch_types={"sem": pltpu.SemaphoreType.DMA(())},
      name="tpu_sync_host_signal_post",
  )
  def send(receipt, sem):
    # Each copy completes before the next starts: the payload lands before
    # the generation that announces it.
    target = target_ref.at[pl.ds(0, payload.shape[0])]
    pltpu.async_copy(payload_ref, target, sem).wait()
    pltpu.async_copy(generation_ref, request_ref, sem).wait()
    pltpu.async_copy(generation_ref, receipt, sem).wait()

  receipt = send()
  return receipt, jax.freeze(target_ref), jax.freeze(request_ref)


def _receive(receipt: jax.Array, response_page: jax.Array) -> jax.Array:
  """Polls until the host answers ``receipt``; returns the answer as a page."""
  receipt_ref = jax.new_ref(receipt)
  response_ref = jax.new_ref(response_page, memory_space=pl.HOST)

  @pl.kernel(
      out_type=pltpu.HBM((PAGE_WORDS,), jnp.int32),
      mesh=_scalar_mesh(),
      compiler_params=_SC_PARAMS,
      scratch_types={
          "expected": pltpu.SMEM((PAGE_WORDS,), jnp.int32),
          "observed": pltpu.SMEM((PAGE_WORDS,), jnp.int32),
          "sem": pltpu.SemaphoreType.DMA(()),
      },
      name="tpu_sync_host_signal_receive",
  )
  def poll_until_answered(status, expected, observed, sem):
    pltpu.async_copy(receipt_ref, expected, sem).wait()
    observed[0] = jnp.int32(0)

    def pending(_):
      answer, generation = observed[0], expected[0]
      return (
          (answer != generation)
          & (answer != -generation)
          & (answer != jnp.int32(STOPPED))
      )

    def poll(_):
      pltpu.async_copy(response_ref, status, sem).wait()
      pltpu.async_copy(status, observed, sem).wait()
      return 0

    lax.while_loop(pending, poll, 0)

  return poll_until_answered()


def _publish(status: jax.Array, retained: jax.Array, words: int) -> jax.Array:
  """``retained[:words]`` in a fresh buffer if ``status[0] > 0``, else zeros."""
  assert 0 < words <= retained.size and words % PAGE_WORDS == 0
  status_ref = jax.new_ref(status, memory_space=pltpu.HBM)
  source_ref = jax.new_ref(retained, memory_space=pltpu.HBM)

  @pl.kernel(
      out_type=pltpu.HBM((words,), retained.dtype),
      mesh=pltpu.TensorCoreMesh(
          axis_name="tpu_sync_host_signal_publish", num_cores=1
      ),
      scratch_types={
          "ready": pltpu.SMEM((PAGE_WORDS,), jnp.int32),
          "staging": pltpu.VMEM((words,), retained.dtype),
          "sem": pltpu.SemaphoreType.DMA(()),
      },
      name="tpu_sync_host_signal_publish",
  )
  def copy(output, ready, staging, sem):
    pltpu.async_copy(status_ref, ready, sem).wait()
    ok = ready[0] > 0

    @pl.when(ok)
    def _():
      pltpu.async_copy(source_ref.at[pl.ds(0, words)], staging, sem).wait()

    @pl.when(jnp.logical_not(ok))
    def _():
      staging[...] = jnp.zeros_like(staging)

    pltpu.async_copy(staging, output, sem).wait()

  return copy()


def wait(
    receipt: jax.Array,
    response_page: jax.Array,
    retained: jax.Array,
    words: int,
) -> jax.Array:
  """Waits for ``receipt``'s answer and returns ``retained[:words]``.

  Args:
    receipt: The HBM receipt ``post`` returned.
    response_page: The channel's response page (pinned host).
    retained: The device buffer the host uploads into (int32).
    words: Result words to read, at most ``retained.size``.

  Returns:
    A fresh ``[words]`` buffer: the host's upload on success, else zeros.
  """
  if not 0 < words <= retained.size:
    raise ValueError(
        f"cannot read {words} words from a {retained.size}-word buffer"
    )
  status = _receive(receipt, response_page)
  return _publish(status, retained, round_pages(words))[:words]
