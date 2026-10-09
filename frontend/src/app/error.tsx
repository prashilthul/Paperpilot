"use client";

import { useEffect } from "react";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    // Log the real cause to the console; render a generic message to the user so
    // backend hostnames and DNS failures are not exposed in the UI.
    console.error(error);
  }, [error]);

  return (
    <div className="flex min-h-[60vh] items-center justify-center px-6">
      <div className="max-w-md text-center">
        <p className="font-mono text-xs uppercase tracking-widest text-muted-foreground">
          Error 500
        </p>
        <h1 className="mt-3 font-serif text-2xl">Something went wrong</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          An unexpected error occurred while loading this page. Please try
          again.
        </p>
        <button
          type="button"
          onClick={reset}
          className="mt-6 inline-flex h-8 items-center justify-center rounded-lg bg-primary px-3 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/80"
        >
          Try again
        </button>
      </div>
    </div>
  );
}