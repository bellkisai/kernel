import type { ReactNode } from "react";
import { AlertCircle, Loader2, type LucideIcon } from "lucide-react";
import { Button } from "./Button";

interface EmptyStateProps {
  icon: LucideIcon;
  heading: string;
  description?: string;
}

export function EmptyState({ icon: Icon, heading, description }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center gap-3 py-8 text-center">
      <Icon size={32} className="text-text-disabled" />
      <h3 className="text-body font-medium text-text-secondary">{heading}</h3>
      {description && (
        <p className="text-caption text-text-muted max-w-xs">{description}</p>
      )}
    </div>
  );
}

interface LoadingStateProps {
  message?: string;
}

export function LoadingState({ message = "Loading..." }: LoadingStateProps) {
  return (
    <div className="flex flex-col items-center gap-3 py-8">
      <Loader2 size={20} className="text-accent animate-spin motion-reduce:animate-none" />
      <p className="text-caption text-text-muted">{message}</p>
    </div>
  );
}

interface ErrorStateProps {
  message: string;
  detail?: string;
  onRetry?: () => void;
  children?: ReactNode;
}

export function ErrorState({ message, detail, onRetry, children }: ErrorStateProps) {
  return (
    <div className="flex flex-col items-center gap-3 py-8 text-center">
      <AlertCircle size={24} className="text-error" />
      <p className="text-body text-error">{message}</p>
      {detail && <p className="text-caption text-text-disabled">{detail}</p>}
      {onRetry && (
        <Button variant="secondary" size="sm" onClick={onRetry}>
          Retry
        </Button>
      )}
      {children}
    </div>
  );
}
