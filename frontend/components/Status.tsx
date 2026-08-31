export function ErrorPanel({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="rounded-lg border border-danger/30 bg-danger/5 p-6 text-sm">
      <p className="font-medium text-danger">Couldn&apos;t load this data</p>
      <p className="mt-1 text-muted">{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-4 rounded border border-accent/40 px-3 py-1.5 text-xs uppercase tracking-wide text-accent transition hover:bg-accent hover:text-accent-foreground"
        >
          Retry
        </button>
      )}
    </div>
  );
}

export function LoadingPanel({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 text-sm text-muted">
      <span className="relative flex h-2 w-2">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent opacity-60" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-accent" />
      </span>
      {label}
    </div>
  );
}

export function EmptyPanel({ message }: { message: string }) {
  return <p className="text-sm text-muted">{message}</p>;
}
