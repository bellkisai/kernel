const FOCUS_RING =
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-canvas";

interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  disabled?: boolean;
}

export function Toggle({ checked, onChange, label, disabled = false }: ToggleProps) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={[
        "relative inline-flex h-5 w-9 shrink-0 rounded-full",
        "transition-colors duration-micro",
        "motion-reduce:transition-none",
        checked ? "bg-accent" : "bg-overlay",
        FOCUS_RING,
        disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer",
      ].join(" ")}
    >
      <span className="sr-only">{label}</span>
      <span
        className={[
          "pointer-events-none inline-block h-4 w-4 rounded-full bg-text-primary shadow-sm",
          "transition-transform duration-micro",
          "motion-reduce:transition-none",
          "translate-y-0.5",
          checked ? "translate-x-[18px]" : "translate-x-0.5",
        ].join(" ")}
      />
    </button>
  );
}
