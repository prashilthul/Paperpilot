"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Upload, FileText, X, AlertCircle } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";
const MAX_FILE_SIZE = 20 * 1024 * 1024; // 20 MB

type Phase = "idle" | "uploading" | "processing" | "complete" | "error";

const PROGRESS_STAGES = [
  { message: "Parsing PDF...", minPercent: 20, maxPercent: 40 },
  { message: "Chunking sections...", minPercent: 40, maxPercent: 65 },
  { message: "Generating embeddings...", minPercent: 65, maxPercent: 90 },
];

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

interface UploadDialogProps {
  trigger?: React.ReactElement;
  buttonText?: string;
  className?: string;
}

export function UploadDialog({ trigger, buttonText = "Upload Paper", className }: UploadDialogProps = {}) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [phase, setPhase] = useState<Phase>("idle");
  const [progressMessage, setProgressMessage] = useState("");
  const [progressPercent, setProgressPercent] = useState(0);
  const [errorMessage, setErrorMessage] = useState("");
  const [paperId, setPaperId] = useState<string | null>(null);
  const [paperTitle, setPaperTitle] = useState("");
  const [dragging, setDragging] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval>>(undefined);
  const progressTimerRef = useRef<ReturnType<typeof setInterval>>(undefined);
  const processingStartRef = useRef(0);

  // Reset state when dialog closes
  const handleOpenChange = (newOpen: boolean) => {
    if (!newOpen) resetState();
    setOpen(newOpen);
  };

  const resetState = () => {
    setPhase("idle");
    setProgressMessage("");
    setProgressPercent(0);
    setErrorMessage("");
    setPaperId(null);
    setPaperTitle("");
    setSelectedFile(null);
    setValidationError("");
    clearInterval(pollRef.current);
    clearInterval(progressTimerRef.current);
  };

  // Simulate progress through parsing/chunking/embedding stages
  useEffect(() => {
    if (phase !== "processing") {
      clearInterval(progressTimerRef.current);
      return;
    }

    processingStartRef.current = Date.now();
    let stageIndex = 0;

    setProgressMessage(PROGRESS_STAGES[0].message);
    setProgressPercent(PROGRESS_STAGES[0].minPercent);

    progressTimerRef.current = setInterval(() => {
      const elapsed = Date.now() - processingStartRef.current;
      const stageDuration = 4000; // ms per stage
      stageIndex = Math.min(
        Math.floor(elapsed / stageDuration),
        PROGRESS_STAGES.length - 1
      );

      const stage = PROGRESS_STAGES[stageIndex];
      setProgressMessage(stage.message);

      // Smooth progress within the stage range
      const stageProgress =
        stageIndex < PROGRESS_STAGES.length - 1
          ? ((elapsed % stageDuration) / stageDuration) * 100
          : 100;
      const percent =
        stage.minPercent +
        (stage.maxPercent - stage.minPercent) * (stageProgress / 100);
      setProgressPercent(Math.min(percent, 90));
    }, 200);

    return () => clearInterval(progressTimerRef.current);
  }, [phase]);

  // Auto-navigate on success
  useEffect(() => {
    if (phase === "complete" && paperId) {
      setProgressPercent(100);
      const timeout = setTimeout(() => {
        router.push(`/papers/${paperId}`);
        setOpen(false);
      }, 1500);
      return () => clearTimeout(timeout);
    }
  }, [phase, paperId, router]);

  const validateFile = useCallback((file: File): string | null => {
    const ext = file.name.split(".").pop()?.toLowerCase();
    if (file.type && file.type !== "application/pdf" && ext !== "pdf") {
      return "Only PDF files are supported";
    }
    if (ext !== "pdf") {
      return "Only PDF files are supported";
    }
    if (file.size > MAX_FILE_SIZE) {
      return `File exceeds 20 MB limit (${formatFileSize(file.size)})`;
    }
    return null;
  }, []);

  const handleFile = useCallback(
    async (file: File) => {
      const validationErr = validateFile(file);
      if (validationErr) {
        setValidationError(validationErr);
        return;
      }

      setValidationError("");
      setSelectedFile(file);
      setPhase("uploading");
      setProgressMessage("Uploading...");
      setProgressPercent(10);

      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch(`${API_BASE}/api/papers/upload`, {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || `Upload failed (HTTP ${res.status})`);
        }

        const data = await res.json();
        setPaperId(data.id);
        setPaperTitle(data.title || file.name);
        setPhase("processing");

        // Start polling for status
        pollRef.current = setInterval(async () => {
          try {
            const statusRes = await fetch(
              `${API_BASE}/api/papers/${data.id}/status`
            );
            if (!statusRes.ok) {
              throw new Error("Status check failed");
            }
            const statusData = await statusRes.json();

            if (statusData.status === "ready") {
              clearInterval(pollRef.current);
              setPhase("complete");
              setProgressMessage("Complete");
            } else if (statusData.status === "error") {
              clearInterval(pollRef.current);
              console.error("Ingestion failed:", statusData.error);
              setPhase("error");
              setErrorMessage("Processing failed. Please try again.");
            }
          } catch {
            clearInterval(pollRef.current);
            setPhase("error");
            setErrorMessage("Status check failed");
          }
        }, 2000);
      } catch (err) {
        // Log the real cause; show a generic message so backend hostnames and
        // raw response bodies are not shown in the dialog.
        console.error(err);
        setPhase("error");
        setErrorMessage("Upload failed. Please try again.");
      }
    },
    [validateFile]
  );

  // Drag handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) handleFile(files[0]);
  };

  const handleBrowseClick = () => {
    inputRef.current?.click();
  };

  const handleRetry = () => {
    resetState();
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      clearInterval(pollRef.current);
      clearInterval(progressTimerRef.current);
    };
  }, []);

  function renderDropZone() {
    if (phase !== "idle" && phase !== "error") {
      return null;
    }

    if (phase === "error") {
      return (
        <div className="flex flex-col items-center gap-3 py-6">
          <AlertCircle className="size-8 text-red-600" />
          <p className="text-sm text-red-600 text-center">{errorMessage}</p>
          <Button variant="outline" size="sm" onClick={handleRetry}>
            Try again
          </Button>
        </div>
      );
    }

    return (
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleBrowseClick}
        className={`flex cursor-pointer flex-col items-center gap-3 rounded-lg border-2 border-dashed p-8 transition-colors ${
          dragging
            ? "border-foreground bg-muted"
            : "border-border hover:border-muted-foreground hover:bg-muted/50"
        }`}
      >
        <Upload className="size-8 text-muted-foreground" />
        <div className="text-center">
          <p className="text-sm font-medium text-foreground">
            Drop your PDF here, or click to browse
          </p>
          <p className="mt-1 text-xs text-muted-foreground">
            PDF files up to 20 MB
          </p>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          className="hidden"
          onChange={handleInputChange}
        />
      </div>
    );
  }

  function renderProgress() {
    if (phase !== "uploading" && phase !== "processing" && phase !== "complete") {
      return null;
    }

    return (
      <div className="space-y-3 py-4 w-full overflow-hidden">
        <div className="flex items-center justify-between gap-2 w-full min-w-0">
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <FileText className="size-4 shrink-0 text-slate-500" />
            <span className="truncate text-xs font-medium text-slate-700 max-w-[240px]">
              {selectedFile?.name}
            </span>
          </div>
          {selectedFile && (
            <span className="shrink-0 text-[11px] font-mono text-slate-500">
              {formatFileSize(selectedFile.size)}
            </span>
          )}
        </div>

        <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
          <div
            className="h-full rounded-full bg-slate-900 transition-all duration-500 ease-out"
            style={{ width: `${progressPercent}%` }}
          />
        </div>

        <div className="flex items-center justify-between text-xs">
          <span className="text-slate-500">
            {progressMessage}
          </span>
          {phase === "complete" && paperTitle && (
            <span className="truncate font-medium text-slate-900 max-w-[180px]">
              {paperTitle}
            </span>
          )}
        </div>
      </div>
    );
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger
        render={
          trigger || (
            <Button
              size="sm"
              className={`bg-slate-900 text-white hover:bg-slate-800 gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold shadow-xs transition-colors cursor-pointer ${className || ""}`}
            >
              <Upload className="size-3.5" />
              <span>{buttonText}</span>
            </Button>
          )
        }
      />
      <DialogContent className="sm:max-w-md w-full overflow-hidden">
        <DialogHeader>
          <DialogTitle>Upload Paper</DialogTitle>
          <DialogDescription>
            Upload a research paper in PDF format for indexing and analysis.
          </DialogDescription>
        </DialogHeader>

        {validationError && (
          <div className="flex items-start gap-2 rounded-lg bg-red-50 p-3 text-sm text-red-700 ring-1 ring-red-200">
            <AlertCircle className="mt-0.5 size-4 shrink-0" />
            <span>{validationError}</span>
            <button
              onClick={() => setValidationError("")}
              className="ml-auto shrink-0 text-red-400 hover:text-red-600"
            >
              <X className="size-4" />
            </button>
          </div>
        )}

        {renderDropZone()}
        {renderProgress()}
      </DialogContent>
    </Dialog>
  );
}
