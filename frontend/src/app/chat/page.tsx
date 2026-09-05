"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Send,
  ChevronDown,
  ChevronRight,
  BookOpen,
  FileText,
  Sparkles,
  Filter,
  X,
  RefreshCw,
  AlertCircle,
  Search,
  Trash2,
  Zap,
  Loader2,
  Plus,
  MessageSquare,
  History,
  PanelLeft,
  PanelLeftClose,
  Square,
  Upload,
  Copy,
  Check,
  ExternalLink,
} from "lucide-react";
import { fetchPapers, fetchPaper, deletePaper, type Paper, type PaperDetail } from "@/lib/api";
import { showToast } from "@/components/toast";
import { ConfirmModal } from "@/components/confirm-modal";
import { TraceWaterfall } from "@/components/trace-waterfall";
import { UploadDialog } from "@/components/upload-dialog";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

// --- Types ---

interface CitationData {
  marker: string;
  paper_id: string;
  chunk_id: string;
  section_heading: string | null;
  snippet?: string;
}

interface TraceSpanData {
  name: string;
  duration_ms: number;
  status: "ok" | "error" | "skipped";
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: CitationData[];
  trace_id?: string;
  trace_spans?: TraceSpanData[];
  streaming?: boolean;
}

// --- Robust SSE event parser ---

async function* streamSSE(
  response: Response
): AsyncGenerator<{ event: string; data: unknown }> {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (value) {
      buffer += decoder.decode(value, { stream: true });
    }

    const parts = buffer.split(/\r?\n\r?\n/);
    buffer = done ? "" : parts.pop() || "";

    for (const part of parts) {
      if (!part.trim()) continue;
      const lines = part.split(/\r?\n/);
      let event = "";
      let dataStr = "";
      for (let line of lines) {
        line = line.trim();
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataStr = line.slice(5).trim();
      }
      if (event && dataStr) {
        try {
          yield { event, data: JSON.parse(dataStr) };
        } catch {
          yield { event, data: dataStr };
        }
      }
    }

    if (done) break;
  }
}

// --- Citation popover component ---

function CitationPopover({
  citation,
  children,
  onInspect,
}: {
  citation: CitationData;
  children: React.ReactNode;
  onInspect?: (heading: string, paperId: string) => void;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (citation.snippet) {
      navigator.clipboard.writeText(citation.snippet);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <Popover>
      <PopoverTrigger className="inline-flex items-center justify-center px-1.5 py-0.5 mx-0.5 text-[11px] font-semibold text-slate-800 bg-slate-100 hover:bg-slate-200 border border-slate-300 rounded shadow-xs transition-all cursor-pointer align-baseline">
        {children}
      </PopoverTrigger>
      <PopoverContent side="top" align="center" className="w-80 sm:w-96 p-3.5 shadow-xl border-slate-200 bg-white">
        <div className="space-y-2.5 text-xs">
          {/* Header */}
          <div className="flex items-center justify-between gap-2 border-b border-slate-100 pb-2">
            <div className="flex items-center gap-1.5 text-slate-900 font-semibold">
              <BookOpen className="size-3.5 text-slate-700" />
              <span>Citation {citation.marker}</span>
            </div>
            <Badge variant="secondary" className="text-[10px] font-medium max-w-[170px] truncate">
              {citation.section_heading || "General Excerpt"}
            </Badge>
          </div>

          {/* Snippet / Excerpt Text */}
          {citation.snippet ? (
            <div className="relative group p-2.5 bg-slate-50 border-l-2 border-slate-900 rounded-r-md text-xs text-slate-700 leading-relaxed font-sans">
              <p className="italic select-text font-serif">
                &ldquo;{citation.snippet}&rdquo;
              </p>
              <button
                onClick={handleCopy}
                title="Copy cited excerpt"
                className="absolute top-1.5 right-1.5 p-1 rounded bg-white/90 border border-slate-200 text-slate-500 hover:text-slate-800 transition-colors shadow-2xs"
              >
                {copied ? <Check className="size-3 text-emerald-600" /> : <Copy className="size-3" />}
              </button>
            </div>
          ) : (
            <p className="text-slate-500 text-[11px]">
              Grounded evidence from: <span className="font-semibold text-slate-700">{citation.section_heading || "Document Body"}</span>
            </p>
          )}

          {/* Action to Jump to Paper Inspector */}
          {onInspect && (
            <div className="pt-1">
              <Button
                variant="outline"
                size="sm"
                className="w-full text-xs h-7 gap-1.5 justify-center font-medium bg-slate-50 hover:bg-slate-100 border-slate-200 text-slate-800"
                onClick={() => onInspect(citation.section_heading || "", citation.paper_id)}
              >
                <ExternalLink className="size-3 text-slate-600" />
                <span>Locate in Paper Inspector →</span>
              </Button>
            </div>
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}

// --- Message content renderer ---

function parseInlineMarkdown(
  text: string,
  citations?: CitationData[],
  onInspect?: (heading: string, paperId: string) => void
): React.ReactNode[] {
  const regex = /(\[\d+\])|(\*\*[^*]+\*\*)|(\*[^*]+\*)|(`[^`]+`)/g;
  const parts: React.ReactNode[] = [];
  let lastIdx = 0;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIdx) {
      parts.push(text.slice(lastIdx, match.index));
    }

    const matchedStr = match[0];
    if (match[1]) {
      const markerNum = parseInt(match[1].slice(1, -1), 10);
      const citation = citations?.find((c) => c.marker === `[${markerNum}]`);
      if (citation) {
        parts.push(
          <CitationPopover key={`cit-${match.index}`} citation={citation} onInspect={onInspect}>
            [{markerNum}]
          </CitationPopover>
        );
      } else {
        parts.push(
          <sup key={`cit-${match.index}`} className="text-xs font-semibold text-slate-700 px-0.5">
            [{markerNum}]
          </sup>
        );
      }
    } else if (match[2]) {
      parts.push(
        <strong key={`b-${match.index}`} className="font-semibold text-slate-900">
          {matchedStr.slice(2, -2)}
        </strong>
      );
    } else if (match[3]) {
      parts.push(
        <em key={`i-${match.index}`} className="italic text-slate-800">
          {matchedStr.slice(1, -1)}
        </em>
      );
    } else if (match[4]) {
      parts.push(
        <code key={`code-${match.index}`} className="rounded bg-slate-100 px-1.5 py-0.5 font-mono text-xs font-medium text-slate-800 border border-slate-200">
          {matchedStr.slice(1, -1)}
        </code>
      );
    }

    lastIdx = match.index + matchedStr.length;
  }

  if (lastIdx < text.length) {
    parts.push(text.slice(lastIdx));
  }

  return parts;
}

function renderAssistantContent(
  content: string,
  citations?: CitationData[],
  onInspect?: (heading: string, paperId: string) => void
) {
  if (!content) return null;

  const lines = content.split("\n");
  const elements: React.ReactNode[] = [];
  let inCodeBlock = false;
  let codeBlockLines: string[] = [];

  lines.forEach((line, idx) => {
    if (line.trim().startsWith("```")) {
      if (inCodeBlock) {
        elements.push(
          <pre key={`block-${idx}`} className="my-2 overflow-x-auto rounded-xl bg-slate-900 p-3.5 font-mono text-xs text-slate-100 border border-slate-800">
            <code>{codeBlockLines.join("\n")}</code>
          </pre>
        );
        codeBlockLines = [];
        inCodeBlock = false;
      } else {
        inCodeBlock = true;
      }
      return;
    }

    if (inCodeBlock) {
      codeBlockLines.push(line);
      return;
    }

    const trimmed = line.trim();
    if (!trimmed) {
      elements.push(<div key={`sp-${idx}`} className="h-1.5" />);
      return;
    }

    if (trimmed.startsWith("### ")) {
      elements.push(
        <h3 key={`h3-${idx}`} className="text-xs font-bold text-slate-900 mt-2 mb-1">
          {parseInlineMarkdown(trimmed.slice(4), citations, onInspect)}
        </h3>
      );
    } else if (trimmed.startsWith("## ")) {
      elements.push(
        <h2 key={`h2-${idx}`} className="text-sm font-bold text-slate-900 mt-2.5 mb-1">
          {parseInlineMarkdown(trimmed.slice(3), citations, onInspect)}
        </h2>
      );
    } else if (trimmed.startsWith("# ")) {
      elements.push(
        <h1 key={`h1-${idx}`} className="text-base font-bold text-slate-900 mt-3 mb-1">
          {parseInlineMarkdown(trimmed.slice(2), citations, onInspect)}
        </h1>
      );
    } else if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
      elements.push(
        <li key={`li-${idx}`} className="ml-4 list-disc text-sm text-slate-700 my-0.5 leading-relaxed">
          {parseInlineMarkdown(trimmed.slice(2), citations, onInspect)}
        </li>
      );
    } else if (/^\d+\.\s/.test(trimmed)) {
      const match = trimmed.match(/^(\d+\.\s)/);
      const prefixLen = match ? match[1].length : 3;
      elements.push(
        <li key={`oli-${idx}`} className="ml-4 list-decimal text-sm text-slate-700 my-0.5 leading-relaxed">
          {parseInlineMarkdown(trimmed.slice(prefixLen), citations, onInspect)}
        </li>
      );
    } else {
      elements.push(
        <p key={`p-${idx}`} className="text-sm leading-relaxed text-slate-700 my-0.5">
          {parseInlineMarkdown(line, citations, onInspect)}
        </p>
      );
    }
  });

  if (inCodeBlock && codeBlockLines.length > 0) {
    elements.push(
      <pre key="block-final" className="my-2 overflow-x-auto rounded-xl bg-slate-900 p-3.5 font-mono text-xs text-slate-100 border border-slate-800">
        <code>{codeBlockLines.join("\n")}</code>
      </pre>
    );
  }

  return <div className="space-y-0.5">{elements}</div>;
}

// --- Pipeline Stage Loading Animation Component ---

const PIPELINE_STAGES = [
  { label: "Embedding Query", detail: "Converting question into 2048d vector space...", icon: Search },
  { label: "Vector Search", detail: "Searching candidate chunks across uploaded papers...", icon: BookOpen },
  { label: "Reranking Context", detail: "Scoring & filtering chunks for maximum precision...", icon: Zap },
  { label: "Generating Answer", detail: "Synthesizing grounded response with citations...", icon: Sparkles },
];

function PipelineLoadingAnimation() {
  const [stageIdx, setStageIdx] = useState(0);

  useEffect(() => {
    const timers = PIPELINE_STAGES.slice(1).map((_, idx) =>
      setTimeout(() => setStageIdx(idx + 1), 1300 + idx * 280 + Math.random() * 300)
    );
    return () => timers.forEach(clearTimeout);
  }, []);

  const currentStage = PIPELINE_STAGES[stageIdx];

  return (
    <div className="py-2 px-1 space-y-4 min-w-[320px]">
      <div className="flex items-center gap-2.5">
        <div className="size-7 rounded-lg bg-slate-900 text-white flex items-center justify-center shrink-0 shadow-xs animate-pulse">
          <currentStage.icon className="size-3.5" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-xs font-bold text-slate-900 tracking-tight">
            {currentStage.label}
          </div>
          <p className="text-[11px] text-slate-500 mt-0.5 font-mono truncate">
            {currentStage.detail}
          </p>
        </div>
        <span className="shrink-0 text-[10px] font-mono text-slate-500">
          {stageIdx + 1}/{PIPELINE_STAGES.length}
        </span>
      </div>

      <div className="flex items-center">
        {PIPELINE_STAGES.map((stage, idx) => {
          const done = idx < stageIdx;
          const active = idx === stageIdx;
          const Icon = stage.icon;
          return (
            <div key={stage.label} className="flex items-center flex-1 last:flex-none">
              <div className="flex flex-col items-center gap-1">
                <div
                  className={`relative size-8 rounded-xl flex items-center justify-center border transition-all duration-500 ${
                    done
                      ? "bg-slate-900 border-slate-900 text-white"
                      : active
                        ? "bg-slate-900 border-slate-900 text-white shadow-lg scale-110 animate-pulse"
                        : "bg-white border-slate-300 text-slate-400"
                  }`}
                >
                  <Icon className="size-3.5" />
                  {active && (
                    <span className="absolute -inset-1 rounded-xl border-2 border-slate-900/30 animate-ping" />
                  )}
                </div>
                <span
                  className={`text-[9px] font-medium ${
                    active ? "text-slate-900" : done ? "text-slate-700" : "text-slate-400"
                  }`}
                >
                  {stage.label.split(" ")[0]}
                </span>
              </div>
              {idx < PIPELINE_STAGES.length - 1 && (
                <div className="flex-1 h-0.5 mx-1 mb-4 rounded-full bg-slate-200 overflow-hidden">
                  <div
                    className={`h-full bg-slate-700 transition-all duration-700 ${
                      idx < stageIdx ? "w-full" : "w-0"
                    }`}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>

      <div className="flex items-center justify-end gap-1.5">
        <Loader2 className="size-3 animate-spin text-slate-500" />
        <span className="text-[10px] font-mono text-slate-400">
          running RAG pipeline
        </span>
      </div>
    </div>
  );
}

// --- Main Chat Page Component ---

export default function ChatPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const initialPaperId = searchParams.get("paper");

  const [availablePapers, setAvailablePapers] = useState<Paper[]>([]);
  const [selectedPaperId, setSelectedPaperId] = useState<string>(initialPaperId || "all");
  const [selectedPaperDetail, setSelectedPaperDetail] = useState<PaperDetail | null>(null);

  const [sectionSearch, setSectionSearch] = useState("");

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [evaluationEnabled, setEvaluationEnabled] = useState(true);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [expandedTraces, setExpandedTraces] = useState<Set<string>>(new Set());

  // Chat session history state
  const [sessions, setSessions] = useState<Array<{ id: string; title: string; created_at: string }>>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [sessionToDelete, setSessionToDelete] = useState<string | null>(null);
  const [loadingSessionId, setLoadingSessionId] = useState<string | null>(null);

  const CHAT_STORAGE_KEY = "paperpilot.active_chat";

  useEffect(() => {
    try {
      const raw = localStorage.getItem(CHAT_STORAGE_KEY);
      if (!raw) return;
      const saved = JSON.parse(raw);
      if (saved.sessionId) setSessionId(saved.sessionId);
      if (Array.isArray(saved.messages)) {
        setMessages(
          saved.messages.map((m: Message) => ({ ...m, streaming: false }))
        );
      }
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    if (!sessionId || messages.length === 0) {
      try {
        localStorage.removeItem(CHAT_STORAGE_KEY);
      } catch {
        // ignore
      }
      return;
    }
    try {
      localStorage.setItem(
        CHAT_STORAGE_KEY,
        JSON.stringify({ sessionId, messages })
      );
    } catch {
      // ignore
    }
  }, [sessionId, messages]);

  const startNewChat = useCallback(() => {
    setSessionId(null);
    setMessages([]);
    setInput("");
    setError(null);
    try {
      localStorage.removeItem(CHAT_STORAGE_KEY);
    } catch {
      // ignore
    }
  }, []);

  const fetchSessions = useCallback(async () => {
    try {
      const base = API_BASE || (typeof window !== "undefined" ? window.location.origin : "");
      const res = await fetch(`${base}/api/chat/sessions`);
      if (res.ok) {
        const data = await res.json();
        const fetchedList: Array<{ id: string; title: string; created_at: string }> = data || [];
        setSessions(fetchedList);

        if (sessionId && !fetchedList.some((s) => s.id === sessionId)) {
          startNewChat();
        }
      }
    } catch {
      // ignore
    }
  }, [sessionId, startNewChat]);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  const loadSession = async (id: string) => {
    if (loadingSessionId || sending) return;
    setLoadingSessionId(id);
    try {
      const base = API_BASE || (typeof window !== "undefined" ? window.location.origin : "");
      const res = await fetch(`${base}/api/chat/sessions/${id}`);
      if (res.ok) {
        const data = await res.json();
        setSessionId(data.id);
        setMessages(
          (data.messages || []).map((m: any) => ({
            id: m.id,
            role: m.role,
            content: m.content,
            citations: m.citations,
            trace_id: m.trace_id,
            streaming: false,
          }))
        );
        setError(null);
      } else {
        throw new Error("Failed to load session");
      }
    } catch {
      showToast("error", "Error", "Failed to load chat history.");
    } finally {
      setLoadingSessionId(null);
    }
  };

  const confirmDeleteSession = async () => {
    if (!sessionToDelete) return;
    const targetId = sessionToDelete;
    setSessionToDelete(null);

    try {
      const base = API_BASE || (typeof window !== "undefined" ? window.location.origin : "");
      const res = await fetch(`${base}/api/chat/sessions/${targetId}`, { method: "DELETE" });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Failed to delete session (HTTP ${res.status})`);
      }

      showToast("success", "Chat Deleted", "Session removed.");

      if (sessionId === targetId) {
        startNewChat();
      }
      await fetchSessions();
    } catch {
      showToast("error", "Error", "Could not delete chat session. Please try again.");
    }
  };

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Auto-resize textarea height as text grows
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  }, [input]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (input.trim() && !sending) {
        sendMessage(input);
      }
    }
  };

  // Load available papers
  const loadPapers = useCallback(async () => {
    try {
      const res = await fetchPapers();
      setAvailablePapers(res.items || []);
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    loadPapers();
  }, [loadPapers]);

  useEffect(() => {
    if (initialPaperId) {
      setSelectedPaperId(initialPaperId);
    }
  }, [initialPaperId]);

  useEffect(() => {
    if (selectedPaperId && selectedPaperId !== "all") {
      fetchPaper(selectedPaperId)
        .then(setSelectedPaperDetail)
        .catch(() => {
          setSelectedPaperDetail(null);
          setSelectedPaperId("all");
          router.replace("/chat");
        });
    } else {
      setSelectedPaperDetail(null);
    }
  }, [selectedPaperId]);

  const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false);

  const handleDeleteActivePaper = () => {
    if (!selectedPaperId || selectedPaperId === "all") return;
    setConfirmDeleteOpen(true);
  };

  const confirmDeleteActivePaper = async () => {
    if (!selectedPaperId || selectedPaperId === "all") return;
    const paperTitle = selectedPaperDetail?.title || "Paper";
    try {
      await deletePaper(selectedPaperId);
      showToast("success", "Paper Deleted", `"${paperTitle}" was removed successfully.`);
      setSelectedPaperId("all");
      setSelectedPaperDetail(null);
      router.push("/chat");
      loadPapers();
    } catch {
      showToast("error", "Delete Failed", "Could not delete paper. Please try again.");
    } finally {
      setConfirmDeleteOpen(false);
    }
  };

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  const handleInspectCitation = useCallback(
    (heading: string, paperId: string) => {
      setInspectorOpen(true);
      if (paperId && paperId !== selectedPaperId) {
        setSelectedPaperId(paperId);
      }
      if (heading) {
        setSectionSearch(heading);
      }
    },
    [selectedPaperId]
  );

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const toggleTrace = (msgId: string) => {
    setExpandedTraces((prev) => {
      const next = new Set(prev);
      if (next.has(msgId)) next.delete(msgId);
      else next.add(msgId);
      return next;
    });
  };

  const sendMessage = useCallback(
    async (query: string) => {
      if (!query.trim() || sending) return;

      setSending(true);
      setError(null);

      const paperIdsFilter =
      selectedPaperId && selectedPaperId !== "all" && availablePapers.some((p) => p.id === selectedPaperId)
        ? [selectedPaperId]
        : undefined;

      const userMsg: Message = {
        id: `user-${Date.now()}`,
        role: "user",
        content: query.trim(),
      };

      const assistantMsg: Message = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: "",
        streaming: true,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setInput("");

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const res = await fetch(`${API_BASE}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: query.trim(),
            paper_ids: paperIdsFilter,
            session_id: sessionId,
            stream: true,
            evaluate: evaluationEnabled,
          }),
          signal: controller.signal,
        });

        if (!res.ok) {
          const errBody = await res.text();
          throw new Error(errBody || `HTTP ${res.status}`);
        }

        let accumulated = "";
        let finalCitations: CitationData[] | undefined;
        let finalTraceId: string | undefined;
        let finalTraceSpans: TraceSpanData[] | undefined;

        for await (const ev of streamSSE(res)) {
          if (ev.event === "token") {
            const d = ev.data as { text?: string };
            if (d.text) {
              accumulated += d.text;
              setMessages((prev) => {
                const copy = [...prev];
                const last = copy[copy.length - 1];
                if (last && last.role === "assistant" && last.streaming) {
                  copy[copy.length - 1] = { ...last, content: accumulated };
                }
                return copy;
              });
            }
          } else if (ev.event === "done") {
            const d = ev.data as {
              citations?: CitationData[];
              trace_id?: string;
              trace_spans?: TraceSpanData[];
              session_id?: string;
            };
            finalCitations = d.citations;
            finalTraceId = d.trace_id;
            finalTraceSpans = d.trace_spans;
            if (d.session_id) setSessionId(d.session_id);
          } else if (ev.event === "error") {
            const d = ev.data as { message?: string };
            throw new Error(d.message || "Unknown error");
          }
        }

        setMessages((prev) => {
          const copy = [...prev];
          const last = copy[copy.length - 1];
          if (last && last.role === "assistant" && last.streaming) {
            copy[copy.length - 1] = {
              ...last,
              content: accumulated,
              citations: finalCitations,
              trace_id: finalTraceId,
              trace_spans: finalTraceSpans,
              streaming: false,
            };
          }
          return copy;
        });
      } catch (err: unknown) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        const msg = err instanceof Error ? err.message : "Request failed";
        setError(msg);
        setMessages((prev) => {
          const copy = [...prev];
          const last = copy[copy.length - 1];
          if (last && last.role === "assistant" && last.streaming) {
            copy[copy.length - 1] = {
              ...last,
              content: last.content || "Sorry, something went wrong while connecting.",
              streaming: false,
            };
          }
          return copy;
        });
      } finally {
        setSending(false);
        abortRef.current = null;
        textareaRef.current?.focus();
        fetchSessions();
      }
    },
    [sending, selectedPaperId, sessionId, evaluationEnabled, availablePapers, fetchSessions]
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    sendMessage(input);
  };

  const handleStop = () => {
    abortRef.current?.abort();
    setSending(false);
    setMessages((prev) => {
      const copy = [...prev];
      const last = copy[copy.length - 1];
      if (last && last.role === "assistant" && last.streaming) {
        copy[copy.length - 1] = {
          ...last,
          content: last.content || "Response stopped.",
          streaming: false,
        };
      }
      return copy;
    });
  };

  const filteredSections = selectedPaperDetail?.sections?.filter((sec) =>
    sec.heading.toLowerCase().includes(sectionSearch.toLowerCase()) ||
    (sec.content && sec.content.toLowerCase().includes(sectionSearch.toLowerCase()))
  ) || [];

  return (
    <div className="flex h-[calc(100vh-3.5rem)] bg-slate-50/50 overflow-hidden">
      {/* Left Chat History Sidebar */}
      {sidebarOpen ? (
        <div className="w-64 border-r border-slate-200 bg-white flex flex-col h-full shrink-0 transition-all">
          <div className="p-3 border-b border-slate-200 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <History className="size-4 text-slate-700" />
              <span className="text-xs font-bold text-slate-900">Chat History</span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setSidebarOpen(false)}
              className="h-7 w-7 p-0 text-slate-500 hover:text-slate-900"
              title="Close sidebar"
            >
              <PanelLeftClose className="size-4" />
            </Button>
          </div>

          <div className="p-2 border-b border-slate-100">
            <Button
              onClick={startNewChat}
              className="w-full h-8 text-xs font-semibold bg-slate-900 text-white hover:bg-slate-800 flex items-center justify-center gap-1.5 shadow-2xs cursor-pointer"
            >
              <Plus className="size-3.5" /> New Chat
            </Button>
          </div>

          <div className="flex-1 overflow-y-auto p-2 space-y-1">
            {sessions.map((sess) => {
              const isActive = sessionId === sess.id;
              const isLoading = loadingSessionId === sess.id;
              return (
                <div
                  key={sess.id}
                  onClick={() => loadSession(sess.id)}
                  className={`group flex items-center justify-between p-2 rounded-lg cursor-pointer text-xs transition-colors ${
                    isActive
                      ? "bg-slate-900 text-white font-semibold shadow-2xs"
                      : "text-slate-700 hover:bg-slate-100 hover:text-slate-900"
                  }`}
                >
                  <div className="flex items-center gap-2 min-w-0 flex-1">
                    {isLoading ? (
                      <Loader2 className="size-3.5 animate-spin text-slate-400 shrink-0" />
                    ) : (
                      <MessageSquare className={`size-3.5 shrink-0 ${isActive ? "text-slate-300" : "text-slate-400"}`} />
                    )}
                    <span className="truncate">{sess.title || "Untitled Chat"}</span>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSessionToDelete(sess.id);
                    }}
                    className={`h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity ${
                      isActive
                        ? "text-slate-300 hover:text-rose-300 hover:bg-slate-800"
                        : "text-slate-400 hover:text-rose-600 hover:bg-rose-50"
                    }`}
                    title="Delete chat"
                  >
                    <Trash2 className="size-3" />
                  </Button>
                </div>
              );
            })}

            {sessions.length === 0 && (
              <p className="text-[11px] text-slate-400 p-3 text-center">No past chats yet.</p>
            )}
          </div>
        </div>
      ) : (
        <div className="p-2 border-r border-slate-200 bg-white flex flex-col shrink-0">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSidebarOpen(true)}
            className="h-8 w-8 p-0 text-slate-600 hover:text-slate-900"
            title="Open Chat History"
          >
            <PanelLeft className="size-4" />
          </Button>
        </div>
      )}

      {/* Main chat column */}
      <div className="flex flex-1 flex-col min-w-0 h-full">
        {/* Top Paper Selection & Context Bar */}
        <div className="border-b border-slate-200 bg-white px-4 py-2.5 shadow-sm flex flex-wrap items-center justify-between gap-3 shrink-0">
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-700">
              <Filter className="size-3.5 text-slate-700" />
              <span>Target Paper:</span>
            </div>
            
            <div className="relative flex-1 max-w-md">
              <select
                value={selectedPaperId}
                onChange={(e) => {
                  const val = e.target.value;
                  setSelectedPaperId(val);
                  if (val === "all") {
                    router.push("/chat");
                  } else {
                    router.push(`/chat?paper=${val}`);
                  }
                }}
                className="w-full h-8 pl-3 pr-8 text-xs font-medium text-slate-800 bg-slate-50 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-slate-400 focus:border-slate-400 truncate cursor-pointer transition-colors"
              >
                <option value="all">All Uploaded Papers (Global RAG Search)</option>
                {availablePapers.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.title || "Untitled Paper"} ({p.year || "N/A"}) • {p.chunk_count} chunks
                  </option>
                ))}
              </select>
            </div>

            {selectedPaperId !== "all" && (
              <div className="flex items-center gap-1">
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 px-2 text-xs text-slate-500 hover:text-slate-900"
                  onClick={() => {
                    setSelectedPaperId("all");
                    router.push("/chat");
                  }}
                >
                  <X className="size-3.5 mr-1" /> Clear filter
                </Button>

                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 w-8 p-0 text-slate-400 hover:text-rose-600 hover:bg-rose-50"
                  title="Delete this paper"
                  onClick={handleDeleteActivePaper}
                >
                  <Trash2 className="size-3.5" />
                </Button>
              </div>
            )}
          </div>

          <div className="flex items-center gap-2">
            <Button
              type="button"
              size="sm"
              variant={evaluationEnabled ? "default" : "outline"}
              onClick={() => setEvaluationEnabled((v) => !v)}
              disabled={sending}
              className={`h-8 text-xs font-medium transition-all ${
                evaluationEnabled
                  ? "bg-slate-900 text-white hover:bg-slate-800"
                  : "text-slate-600 border-slate-300"
              }`}
            >
              <Sparkles className="size-3 mr-1.5 text-amber-400" />
              {evaluationEnabled ? "Eval On" : "Eval Off"}
            </Button>

            <Button
              type="button"
              size="sm"
              variant={inspectorOpen ? "default" : "outline"}
              onClick={() => setInspectorOpen((v) => !v)}
              className={`h-8 text-xs font-medium transition-all ${
                inspectorOpen
                  ? "bg-slate-900 text-white hover:bg-slate-800"
                  : "text-slate-600 border-slate-300"
              }`}
              title={inspectorOpen ? "Hide Paper Inspector" : "Show Paper Inspector"}
            >
              <BookOpen className="size-3 mr-1.5 text-slate-300" />
              {inspectorOpen ? "Inspector On" : "Inspector Off"}
            </Button>

            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={loadPapers}
              className="h-8 w-8 p-0 text-slate-600 border-slate-300"
              title="Refresh paper list"
            >
              <RefreshCw className="size-3.5" />
            </Button>
          </div>
        </div>

        {/* Message Stream Scroll Area */}
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
          {messages.length === 0 && !error && (
            <div className="flex flex-col items-center justify-center min-h-[70%] text-center px-4 py-8 max-w-xl mx-auto">
              <div className="size-12 rounded-2xl bg-slate-100 border border-slate-200 flex items-center justify-center mb-4 shadow-sm">
                <Sparkles className="size-6 text-slate-800" />
              </div>
              <h2 className="font-serif text-2xl font-semibold text-slate-900 mb-2">
                Paper Pilot Q&A Assistant
              </h2>
              <p className="text-sm text-slate-600 leading-relaxed mb-6">
                Ask targeted questions about your research papers. Get grounded responses backed by exact section citations and trace inspection.
              </p>

              {/* Active Selection Pill */}
              {selectedPaperDetail ? (
                <div className="w-full bg-white border border-slate-200 rounded-xl p-4 mb-6 shadow-sm text-left">
                  <div className="flex items-start justify-between gap-2 mb-2">
                    <span className="text-xs font-semibold uppercase tracking-wider text-slate-700">
                      Active Filtered Paper
                    </span>
                    <Badge variant="secondary" className="text-[10px]">
                      {selectedPaperDetail.chunk_count} Chunks Loaded
                    </Badge>
                  </div>
                  <h3 className="text-sm font-bold text-slate-900 leading-snug">
                    {selectedPaperDetail.title}
                  </h3>
                  <p className="text-xs text-slate-500 mt-1">
                    {selectedPaperDetail.authors?.length
                      ? selectedPaperDetail.authors.join(", ")
                      : "Authors available in PDF"}
                  </p>
                </div>
              ) : (
                <div className="w-full bg-white border border-slate-200 rounded-xl p-3 mb-6 shadow-sm text-xs text-slate-600 flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    <BookOpen className="size-4 text-slate-400" />
                    Searching across <strong>{availablePapers.length} uploaded papers</strong>
                  </span>
                  <span className="text-[11px] text-slate-700 font-medium">Global Mode</span>
                </div>
              )}

              {/* Quick Prompt Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 w-full">
                <button
                  onClick={() => sendMessage("Summarize the key contributions and methodology of this paper.")}
                  className="p-3 bg-white border border-slate-200 hover:border-slate-400 hover:shadow-md rounded-xl text-left transition-all group"
                >
                  <p className="text-xs font-semibold text-slate-900 group-hover:text-slate-800 flex items-center gap-1.5">
                    <Sparkles className="size-3.5 text-slate-700" />
                    <span>Key Contributions</span>
                  </p>
                  <p className="text-[11px] text-slate-500 mt-1 line-clamp-2">
                    Summarize methodology & key findings
                  </p>
                </button>

                <button
                  onClick={() => sendMessage("What datasets, experimental setup, or baselines were evaluated?")}
                  className="p-3 bg-white border border-slate-200 hover:border-slate-400 hover:shadow-md rounded-xl text-left transition-all group"
                >
                  <p className="text-xs font-semibold text-slate-900 group-hover:text-slate-800 flex items-center gap-1.5">
                    <FileText className="size-3.5 text-slate-700" />
                    <span>Datasets & Setup</span>
                  </p>
                  <p className="text-[11px] text-slate-500 mt-1 line-clamp-2">
                    Inspect evaluation metrics & data
                  </p>
                </button>

                <button
                  onClick={() => sendMessage("What are the main results, conclusions, and limitations?")}
                  className="p-3 bg-white border border-slate-200 hover:border-slate-400 hover:shadow-md rounded-xl text-left transition-all group"
                >
                  <p className="text-xs font-semibold text-slate-900 group-hover:text-slate-800 flex items-center gap-1.5">
                    <BookOpen className="size-3.5 text-slate-700" />
                    <span>Main Results</span>
                  </p>
                  <p className="text-[11px] text-slate-500 mt-1 line-clamp-2">
                    Review conclusions & limitations
                  </p>
                </button>
              </div>
            </div>
          )}

          {/* Render Messages */}
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              {msg.role === "user" ? (
                <div className="max-w-[80%] sm:max-w-[70%] rounded-2xl rounded-tr-xs bg-slate-900 text-white px-4 py-3 shadow-sm">
                  <p className="whitespace-pre-wrap text-sm leading-relaxed">{msg.content}</p>
                </div>
              ) : (
                <div className="max-w-[90%] sm:max-w-[80%] flex gap-3">
                  <div className="size-8 rounded-xl bg-slate-900 text-white flex items-center justify-center shrink-0 shadow-sm mt-0.5">
                    <Sparkles className="size-4" />
                  </div>
                  <div className="flex-1 bg-white border border-slate-200 rounded-2xl rounded-tl-xs p-4 shadow-sm text-slate-800">
                    {msg.streaming && !msg.content ? (
                      <PipelineLoadingAnimation />
                    ) : (
                      renderAssistantContent(msg.content, msg.citations, handleInspectCitation)
                    )}

                    {msg.streaming && msg.content && (
                      <span className="inline-block w-1.5 h-4 bg-slate-600 animate-pulse ml-1 align-text-bottom rounded" />
                    )}

                    {/* Trace Waterfall Accordion */}
                    {!msg.streaming && (msg.trace_id || msg.trace_spans) && (
                      <div className="mt-4 pt-3 border-t border-slate-100">
                        <button
                          onClick={() => toggleTrace(msg.id)}
                          className="flex items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-900 transition-colors"
                        >
                          {expandedTraces.has(msg.id) ? (
                            <ChevronDown className="size-3.5 text-slate-800" />
                          ) : (
                            <ChevronRight className="size-3.5" />
                          )}
                          <span>Pipeline Execution Trace</span>
                        </button>
                        {expandedTraces.has(msg.id) && (
                          <div className="mt-2.5">
                            <TraceWaterfall spans={msg.trace_spans || []} />
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}

          {error && !sending && (
            <div className="flex justify-center my-4">
              <div className="bg-rose-50 text-rose-700 text-xs font-medium rounded-xl px-4 py-2.5 border border-rose-200 shadow-sm flex items-center gap-2">
                <AlertCircle className="size-4 text-rose-500" />
                <span>{error}</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div className="border-t border-slate-200 bg-white px-4 py-3 shadow-sm shrink-0">
          <form onSubmit={handleSubmit} className="flex items-end gap-2 max-w-4xl mx-auto">
            <UploadDialog
              trigger={
                <Button
                  type="button"
                  size="icon"
                  variant="outline"
                  aria-label="Upload paper"
                  title="Upload new research paper"
                  className="h-11 w-11 rounded-xl border-slate-300 text-slate-700 hover:bg-slate-100 shadow-2xs shrink-0 transition-all cursor-pointer"
                >
                  <Upload className="size-4 text-slate-700" />
                </Button>
              }
            />

            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                selectedPaperDetail
                  ? `Ask about "${selectedPaperDetail.title.slice(0, 30)}..." (Shift+Enter for newline)`
                  : "Ask a question... (Enter to send, Shift+Enter for newline)"
              }
              disabled={sending}
              className="flex-1 min-h-[44px] max-h-40 py-2.5 px-4 text-sm bg-slate-50 border border-slate-300 rounded-xl focus:bg-white focus:outline-none focus:ring-2 focus:ring-slate-400 focus:border-slate-400 transition-all resize-none font-sans leading-relaxed text-slate-900 placeholder:text-slate-400"
            />

            <Button
              type="submit"
              size="icon"
              disabled={!input.trim() || sending}
              className="h-11 w-11 rounded-xl bg-slate-900 hover:bg-slate-800 text-white shadow-2xs shrink-0 transition-all disabled:opacity-50 cursor-pointer"
            >
              <Send className="size-4" />
            </Button>
            {sending && (
              <Button
                type="button"
                size="icon"
                onClick={handleStop}
                aria-label="Stop generating"
                className="h-11 w-11 rounded-xl bg-rose-600 hover:bg-rose-700 text-white shadow-2xs shrink-0 transition-all cursor-pointer"
              >
                <Square className="size-4" />
              </Button>
            )}
          </form>
        </div>
      </div>

      {/* Right Drawer — Paper Reader & Document Sections */}
      {inspectorOpen && (
        <div className="hidden lg:flex w-96 border-l border-slate-200 bg-white flex-col shrink-0 h-full">
          {/* Header */}
          <div className="border-b border-slate-200 px-4 py-3 bg-slate-50 flex items-center justify-between shrink-0">
            <h3 className="text-xs font-bold text-slate-800 uppercase tracking-wider flex items-center gap-1.5">
              <BookOpen className="size-3.5 text-slate-700" />
              Paper Inspector
            </h3>

            <div className="flex items-center gap-2">
              {selectedPaperDetail && (
                <Badge variant="outline" className="text-[10px] font-mono">
                  {selectedPaperDetail.chunk_count} Chunks
                </Badge>
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setInspectorOpen(false)}
                className="h-6 w-6 p-0 text-slate-400 hover:text-slate-900 hover:bg-slate-200 rounded-md"
                title="Close Paper Inspector"
              >
                <X className="size-3.5" />
              </Button>
            </div>
          </div>

          {/* Drawer Content */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {selectedPaperDetail ? (
              <>
                {/* Paper Title Summary Card */}
                <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-xl space-y-1.5">
                  <div className="flex items-start justify-between gap-2">
                    <h4 className="text-xs font-bold text-slate-900 leading-snug">
                      {selectedPaperDetail.title}
                    </h4>
                  </div>
                  <p className="text-[11px] text-slate-500">
                    {selectedPaperDetail.authors?.length > 0
                      ? selectedPaperDetail.authors.join(", ")
                      : "Authors not specified"}
                  </p>
                </div>

                {/* Abstract Section */}
                {selectedPaperDetail.abstract && (
                  <div className="space-y-1.5">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                      Abstract
                    </span>
                    <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-600 leading-relaxed max-h-48 overflow-y-auto">
                      {selectedPaperDetail.abstract}
                    </div>
                  </div>
                )}

                {/* Document Sections List */}
                <div className="space-y-2 pt-2 border-t border-slate-100">
                  <div className="flex items-center justify-between">
                    <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                      Document Sections ({selectedPaperDetail.sections?.length || 0})
                    </span>
                  </div>

                  <div className="relative">
                    <Search className="absolute left-2.5 top-1/2 size-3 -translate-y-1/2 text-slate-400" />
                    <Input
                      placeholder="Filter sections or text..."
                      value={sectionSearch}
                      onChange={(e) => setSectionSearch(e.target.value)}
                      className="h-7 pl-8 text-xs bg-slate-50"
                    />
                  </div>

                  <div className="space-y-1.5 max-h-[calc(100vh-24rem)] overflow-y-auto pr-1">
                    {filteredSections.map((sec) => (
                      <details
                        key={sec.id}
                        className="group rounded-lg border border-slate-200 bg-white text-xs overflow-hidden transition-colors"
                      >
                        <summary className="flex items-center justify-between p-2.5 cursor-pointer font-semibold text-slate-800 hover:bg-slate-50 transition-colors select-none">
                          <span className="truncate pr-2">{sec.heading}</span>
                          <ChevronRight className="size-3.5 text-slate-400 group-open:rotate-90 transition-transform shrink-0" />
                        </summary>
                        {sec.content && (
                          <div className="p-2.5 pt-0 text-slate-600 border-t border-slate-100 bg-slate-50/50 leading-relaxed text-[11px] max-h-40 overflow-y-auto">
                            {sec.content}
                          </div>
                        )}
                      </details>
                    ))}
                    {filteredSections.length === 0 && (
                      <p className="text-center py-4 text-xs text-slate-400">
                        No matching sections found.
                      </p>
                    )}
                  </div>
                </div>
              </>
            ) : (
              <div className="flex flex-col items-center justify-center h-full text-center p-6 space-y-3">
                <div className="size-12 rounded-2xl bg-slate-100 flex items-center justify-center text-slate-400">
                  <FileText className="size-6" />
                </div>
                <div>
                  <p className="text-xs font-semibold text-slate-800">No Active Paper Selected</p>
                  <p className="text-[11px] text-slate-500 mt-1">
                    Select a specific paper from the top dropdown or upload a PDF to inspect its full document sections.
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal for Paper */}
      <ConfirmModal
        open={confirmDeleteOpen}
        onOpenChange={setConfirmDeleteOpen}
        title="Delete Paper?"
        description={`Are you sure you want to delete "${selectedPaperDetail?.title || "this paper"}"? All associated sections and embeddings will be permanently removed.`}
        confirmText="Delete Paper"
        variant="destructive"
        onConfirm={confirmDeleteActivePaper}
      />

      {/* Delete Confirmation Modal for Chat Session */}
      <ConfirmModal
        open={!!sessionToDelete}
        onOpenChange={(open) => { if (!open) setSessionToDelete(null); }}
        title="Delete Chat Session?"
        description="Are you sure you want to delete this chat session and its conversation history?"
        confirmText="Delete Session"
        variant="destructive"
        onConfirm={confirmDeleteSession}
      />
    </div>
  );
}
