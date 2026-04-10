import { type ButtonHTMLAttributes, forwardRef } from "react";
import type { LucideIcon } from "lucide-react";

const FOCUS_RING =
  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-canvas";

const sizes = {
  sm: { button: "w-7 h-7", icon: 14 },
  md: { button: "w-8 h-8", icon: 16 },
} as const;

interface IconButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "children"> {
  icon: LucideIcon;
  size?: keyof typeof sizes;
  tooltip: string;
  active?: boolean;
}

export const IconButton = forwardRef<HTMLButtonElement, IconButtonProps>(
  ({ icon: Icon, size = "sm", tooltip, active = false, disabled, className = "", ...props }, ref) => {
    const s = sizes[size];
    return (
      <button
        ref={ref}
        title={tooltip}
        aria-label={tooltip}
        disabled={disabled}
        className={[
          "inline-flex items-center justify-center rounded",
          "transition-colors duration-micro",
          "motion-reduce:transition-none",
          s.button,
          active
            ? "bg-accent/20 text-accent"
            : "text-text-muted hover:bg-overlay hover:text-text-secondary",
          FOCUS_RING,
          disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer",
          className,
        ].join(" ")}
        {...props}
      >
        <Icon size={s.icon} />
      </button>
    );
  },
);

IconButton.displayName = "IconButton";
