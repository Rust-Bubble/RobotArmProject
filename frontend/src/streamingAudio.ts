function abortError() {
  return new DOMException("播放已取消", "AbortError");
}

function waitForEvent(
  target: EventTarget,
  eventName: string,
  signal: AbortSignal,
) {
  return new Promise<void>((resolve, reject) => {
    const cleanup = () => {
      target.removeEventListener(eventName, onEvent);
      signal.removeEventListener("abort", onAbort);
    };
    const onEvent = () => {
      cleanup();
      resolve();
    };
    const onAbort = () => {
      cleanup();
      reject(abortError());
    };

    target.addEventListener(eventName, onEvent, { once: true });
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

async function appendChunk(
  sourceBuffer: SourceBuffer,
  chunk: Uint8Array,
  signal: AbortSignal,
) {
  if (signal.aborted) throw abortError();
  const buffer = new Uint8Array(chunk).buffer;
  sourceBuffer.appendBuffer(buffer);
  await waitForEvent(sourceBuffer, "updateend", signal);
}

export type PlaybackHandle = {
  objectUrl: string;
  streamed: boolean;
};

export async function playAudioResponse(
  response: Response,
  audio: HTMLAudioElement,
  signal: AbortSignal,
): Promise<PlaybackHandle> {
  const contentType =
    response.headers.get("content-type")?.split(";", 1)[0].trim() ||
    "audio/mpeg";
  const canStream =
    Boolean(response.body) &&
    typeof MediaSource !== "undefined" &&
    MediaSource.isTypeSupported(contentType);

  if (!canStream) {
    const blob = await response.blob();
    if (signal.aborted) throw abortError();
    const objectUrl = URL.createObjectURL(blob);
    audio.src = objectUrl;
    await audio.play();
    return { objectUrl, streamed: false };
  }

  const mediaSource = new MediaSource();
  const objectUrl = URL.createObjectURL(mediaSource);
  audio.src = objectUrl;

  try {
    await waitForEvent(mediaSource, "sourceopen", signal);
    const sourceBuffer = mediaSource.addSourceBuffer(contentType);
    const reader = response.body!.getReader();
    let playPromise: Promise<void> | null = null;

    while (true) {
      if (signal.aborted) throw abortError();
      const { done, value } = await reader.read();
      if (done) break;
      if (!value.byteLength) continue;
      await appendChunk(sourceBuffer, value, signal);
      playPromise ??= audio.play();
    }

    if (!playPromise) throw new Error("TTS 服务返回了空音频");
    if (mediaSource.readyState === "open") mediaSource.endOfStream();
    await playPromise;
    return { objectUrl, streamed: true };
  } catch (error) {
    URL.revokeObjectURL(objectUrl);
    throw error;
  }
}
