"use client";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="studio-shell">
      <main className="main">
        <div className="content">
          <div className="page-head">
            <div>
              <div className="eyebrow">Studio error</div>
              <h1>We could not load this workspace</h1>
              <p className="subtitle">
                The Studio UI hit an unexpected error. Your project files and
                trained artifacts are not modified by this screen.
              </p>
            </div>
          </div>

          <div className="card empty-state">
            <strong>Try again</strong>
            <span>{error?.message || "Unexpected Studio error."}</span>
            <button className="primary" onClick={() => reset()}>
              Reload workspace
            </button>
          </div>
        </div>
      </main>
    </div>
  );
}
