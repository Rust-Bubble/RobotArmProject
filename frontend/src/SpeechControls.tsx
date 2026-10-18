import { useEffect, useRef, useState } from "react";
import { LoaderCircle, Mic, Volume2 } from "lucide-react";
import { playAudioResponse } from "./streamingAudio";
const TYPES = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
function getType() { if (typeof MediaRecorder === "undefined") return ""; return TYPES.find((t) => MediaRecorder.isTypeSupported(t)) ?? ""; }
function filename(t: string) { return t.includes("mp4") ? "recording.mp4" : t.includes("ogg") ? "recording.ogg" : "recording.webm"; }
async function apiError(r: Response, label: string) {
  try { const body = (await r.json()) as { detail?: string }; return body.detail || label + "（" + r.status + "）"; }
  catch { return label + "（" + r.status + "）"; }
}
export function VoiceInput({ disabled, onTranscript }: { disabled: boolean; onTranscript: (text: string) => void }) {
  const [recording, setRecording] = useState(false), [busy, setBusy] = useState(false), [error, setError] = useState("");
  const recorder = useRef<MediaRecorder | null>(null), stream = useRef<MediaStream | null>(null);
  const chunks = useRef<Blob[]>([]), request = useRef<AbortController | null>(null);
  const stopTracks = () => { stream.current?.getTracks().forEach((t) => t.stop()); stream.current = null; };
  useEffect(() => () => {
    request.current?.abort();
    if (recorder.current?.state === "recording") { recorder.current.onstop = null; recorder.current.stop(); }
    stream.current?.getTracks().forEach((t) => t.stop());
  }, []);
  const transcribe = async (blob: Blob, type: string) => {
    if (!blob.size) { setError("未录到有效音频，请重试。"); setBusy(false); return; }
    const controller = new AbortController(); request.current = controller;
    const form = new FormData(); form.append("file", blob, filename(type)); setBusy(true);
    try {
      const response = await fetch("/api/stt/transcribe", { method: "POST", body: form, signal: controller.signal });
      if (!response.ok) throw new Error(await apiError(response, "转写失败"));
      const body = (await response.json()) as { text?: unknown };
      if (typeof body.text !== "string" || !body.text.trim()) throw new Error("STT 服务未返回转写文本。");
      setError(""); onTranscript(body.text.trim());
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") return;
      setError(e instanceof Error ? e.message : "转写失败，请重试。");
    } finally { if (request.current === controller) request.current = null; setBusy(false); }
  };
  const start = async () => {
    setError("");
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") { setError("当前浏览器不支持麦克风录入。"); return; }
    try {
      const media = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
      const type = getType(), instance = new MediaRecorder(media, type ? { mimeType: type } : undefined);
      stream.current = media; recorder.current = instance; chunks.current = [];
      instance.ondataavailable = (e) => { if (e.data.size) chunks.current.push(e.data); };
      instance.onerror = () => { setError("录音过程中发生错误，请重试。"); setRecording(false); stopTracks(); };
      instance.onstop = () => {
        const resolved = instance.mimeType || type || "audio/webm", blob = new Blob(chunks.current, { type: resolved });
        chunks.current = []; recorder.current = null; void transcribe(blob, resolved);
      };
      instance.start(250); setRecording(true);
    } catch (e) {
      stopTracks();
      const denied = e instanceof DOMException && (e.name === "NotAllowedError" || e.name === "SecurityError");
      setError(denied ? "未获得麦克风权限，请允许访问后重试。" : "无法启动麦克风，请检查设备后重试。");
    }
  };
  const toggle = () => { if (recording) { setRecording(false); stopTracks(); recorder.current?.stop(); } else void start(); };
  return <span className="voice-input-control">
    <button type="button" className={"mic-button small" + (recording ? " recording" : "")} onClick={toggle} disabled={disabled || busy} aria-label={recording ? "结束语音录入" : "语音输入"}>
      {busy ? <LoaderCircle className="spin" size={18} /> : <Mic size={18} />}
    </button>
    {error && <span className="voice-input-error" role="status">{error}</span>}
  </span>;
}
export function SpeakButton({ text }: { text: string }) {
  const [state, setState] = useState<"idle" | "loading" | "playing">("idle");
  const controller = useRef<AbortController | null>(null), audio = useRef<HTMLAudioElement | null>(null), url = useRef<string | null>(null);
  useEffect(() => () => { controller.current?.abort(); audio.current?.pause(); if (url.current) URL.revokeObjectURL(url.current); }, []);
  const speak = async () => {
    controller.current?.abort(); audio.current?.pause(); if (url.current) URL.revokeObjectURL(url.current);
    const abort = new AbortController(), player = new Audio(); controller.current = abort; audio.current = player; setState("loading");
    player.addEventListener("playing", () => setState("playing"), { once: true });
    player.addEventListener("ended", () => setState("idle"), { once: true });
    try {
      const response = await fetch("/api/tts/speech", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }), signal: abort.signal });
      if (!response.ok) throw new Error(await apiError(response, "语音合成失败"));
      const result = await playAudioResponse(response, player, abort.signal); url.current = result.objectUrl;
    } catch (e) { if (e instanceof DOMException && e.name === "AbortError") return; setState("idle"); console.error(e); }
  };
  return <button type="button" className="tts-button" onClick={() => void speak()} disabled={state === "loading"}>
    {state === "loading" ? <LoaderCircle className="spin" size={14} /> : <Volume2 size={14} />}
    {state === "loading" ? "正在合成" : state === "playing" ? "正在播放" : "播放语音"}
  </button>;
}
